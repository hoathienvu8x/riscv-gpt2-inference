#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

/*
 *  ABLATION CONFIGURATION
 *  1 = Use RISC-V Vector (RVV) Implementation
 *  0 = Use Scalar Baseline Implementation
 */

#define ENABLE_RVV_MATMUL     0 
#define ENABLE_RVV_LAYERNORM  0 
#define ENABLE_RVV_ADD        0

#if (ENABLE_RVV_MATMUL || ENABLE_RVV_LAYERNORM || ENABLE_RVV_ADD )
  #include <riscv_vector.h>
  #define HAS_RVV_HEADER 1
#endif

#define N_LAYERS 12
#define D_MODEL 768
#define N_HEADS 12
#define HEAD_SIZE 64   /* 768 / 12 */
#define VOCAB_SIZE 50257
#define MAX_SEQ_LEN 1024
#define EPS 1e-5f

#define SQRT_2_PI 0.7978845608f
#define C_GELU 0.044715f

typedef struct {
  float *ln1_w, *ln1_b;
  float *attn_w, *attn_b;
  float *attn_proj_w, *attn_proj_b;
  float *ln2_w, *ln2_b;
  float *mlp_fc_w, *mlp_fc_b;
  float *mlp_proj_w, *mlp_proj_b;
} GPT2Layers;

typedef struct {
  float *wte; /* [VOCAB, D_MODEL] */
  float *wpe; /* [MAX_SEQ, D_MODEL] */
  GPT2Layers layers[N_LAYERS];
  float *ln_f_w, *ln_f_b;
  float *lm_head;
} GPT2Weights;

typedef struct {
  float *key_cache;   /* [Layers * Heads * Seq * HeadSize] */
  float *value_cache; /* [Layers * Heads * Seq * HeadSize] */

  float *x;
  float *logits;
  float *final;

  int seq_len;

  float *qkv;
  float *att_scores;
  float *attn_out;

  float *resid;
  float *ln1_out;
  float *ln2_out;
  float *mlp_hidden;
  float *mlp_out;
} GPT2State;


/* SCALAR KERNELS (BASELINE) */
void add(float *out, float *a, float *b, int size) {
  #ifdef HAS_RVV_HEADER
  size_t vl;
  for (int i = 0; i < size; i += vl) {
    vl = __riscv_vsetvl_e32m8(size - i);
    vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vl);
    vfloat32m8_t vb = __riscv_vle32_v_f32m8(b + i, vl);
    vfloat32m8_t vres = __riscv_vfadd_vv_f32m8(va, vb, vl);
    __riscv_vse32_v_f32m8(out + i, vres, vl);
  }
  #else
  for(int i=0; i<size; i++) out[i] = a[i] + b[i];
  #endif
}

void gelu(float *x, int size) {
  for(int i=0; i<size; i++) {
    float xv = x[i];
    float cube = C_GELU * xv * xv * xv;
    float inner = SQRT_2_PI * (xv + cube);
    x[i] = 0.5f * xv * (1.0f + tanhf(inner));
  }
}

void layernorm(float *out, float *x, float *g, float *b, int size) {
  #ifdef HAS_RVV_HEADER
  size_t vl;
  /* Mean */
  vfloat32m1_t v_sum = __riscv_vfmv_v_f_f32m1(0.0f, 1);
  int ptr = 0;
  while(ptr < size) {
    vl = __riscv_vsetvl_e32m8(size - ptr);
    vfloat32m8_t v_data = __riscv_vle32_v_f32m8(x + ptr, vl);
    v_sum = __riscv_vfredusum_vs_f32m8_f32m1(v_data, v_sum, vl);
    ptr += vl;
  }
  float mean = __riscv_vfmv_f_s_f32m1_f32(v_sum) / size;

  /* Variance */
  vfloat32m1_t v_var = __riscv_vfmv_v_f_f32m1(0.0f, 1);
  ptr = 0;
  while(ptr < size) {
    vl = __riscv_vsetvl_e32m8(size - ptr);
    vfloat32m8_t v_data = __riscv_vle32_v_f32m8(x + ptr, vl);
    vfloat32m8_t v_diff = __riscv_vfsub_vf_f32m8(v_data, mean, vl);
    vfloat32m8_t v_sq = __riscv_vfmul_vv_f32m8(v_diff, v_diff, vl);
    v_var = __riscv_vfredusum_vs_f32m8_f32m1(v_sq, v_var, vl);
    ptr += vl;
  }
  float var = __riscv_vfmv_f_s_f32m1_f32(v_var) / size;
  float inv_std = 1.0f / sqrtf(var + EPS);

  /* Normalize */
  ptr = 0;
  while(ptr < size) {
    vl = __riscv_vsetvl_e32m8(size - ptr);
    vfloat32m8_t v_data = __riscv_vle32_v_f32m8(x + ptr, vl);
    vfloat32m8_t v_g = __riscv_vle32_v_f32m8(g + ptr, vl);
    vfloat32m8_t v_b = __riscv_vle32_v_f32m8(b + ptr, vl);
    
    vfloat32m8_t v_norm = __riscv_vfsub_vf_f32m8(v_data, mean, vl);
    v_norm = __riscv_vfmul_vf_f32m8(v_norm, inv_std, vl);
    v_norm = __riscv_vfmacc_vv_f32m8(v_b, v_norm, v_g, vl);
    
    __riscv_vse32_v_f32m8(out + ptr, v_norm, vl);
    ptr += vl;
  }
  #else
  float mean = 0.0f;
  for(int i=0; i<size; i++) mean += x[i];
  mean /= size;
  
  float var = 0.0f;
  for(int i=0; i<size; i++) var += (x[i] - mean) * (x[i] - mean);
  var /= size;
  
  float inv_std = 1.0f / sqrtf(var + EPS);
  for(int i=0; i<size; i++) {
    out[i] = (x[i] - mean) * inv_std * g[i] + b[i];
  }
  #endif
}

void matmul(float *out, float *x, float *w, float *b, int dim_in, int dim_out) {
  #ifdef HAS_RVV_HEADER
  size_t vl;
  for (int i = 0; i < dim_out; i += vl) {
    vl = __riscv_vsetvl_e32m8(dim_out - i);
    vfloat32m8_t v_acc;
    if (b != NULL) v_acc = __riscv_vle32_v_f32m8(b + i, vl);
    else v_acc = __riscv_vfmv_v_f_f32m8(0.0f, vl);

    for (int j = 0; j < dim_in; j++) {
      float scalar_x = x[j];
      vfloat32m8_t v_w = __riscv_vle32_v_f32m8(w + (j * dim_out + i), vl);
      v_acc = __riscv_vfmacc_vf_f32m8(v_acc, scalar_x, v_w, vl);
    }
    __riscv_vse32_v_f32m8(out + i, v_acc, vl);
  }
  #else
  for (int i = 0; i < dim_out; i++) {
    float val = (b != NULL) ? b[i] : 0.0f;
    for (int j = 0; j < dim_in; j++) {
      val += x[j] * w[j * dim_out + i];
    }
    out[i] = val;
  }
  #endif
}

/* Mapping */
#if ENABLE_RVV_MATMUL
  const char* MODE_MATMUL = "RVV";
#else
  const char* MODE_MATMUL = "Scalar";
#endif

#if ENABLE_RVV_LAYERNORM
  const char* MODE_LN = "RVV";
#else
  const char* MODE_LN = "Scalar";
#endif

#if ENABLE_RVV_ADD
  const char* MODE_ADD = "RVV";
#else
  const char* MODE_ADD = "Scalar";
#endif


void softmax(float *x, int n) {
  float max_val = x[0];
  for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  for (int i = 0; i < n; i++) x[i] /= sum;
}

void attention(float *out, float *x, 
               float *c_attn_w, float *c_attn_b, 
               float *c_proj_w, float *c_proj_b,
               GPT2State *state, int layer, int pos) {
    
  /* 1. QKV Projection (Mapped) */
  matmul(state->qkv, x, c_attn_w, c_attn_b, D_MODEL, 3 * D_MODEL);

  float *q = state->qkv;
  float *k = state->qkv + D_MODEL;
  float *v = state->qkv + 2 * D_MODEL;

  /* 2. Multi-Head Attention Loop */
  for (int h = 0; h < N_HEADS; h++) {
    float *head_q = q + h * HEAD_SIZE;
    
    int cache_offset = layer * (N_HEADS * MAX_SEQ_LEN * HEAD_SIZE) + 
                       h * (MAX_SEQ_LEN * HEAD_SIZE) + 
                       pos * HEAD_SIZE;
    
    float *cache_k = state->key_cache + cache_offset;
    float *cache_v = state->value_cache + cache_offset;

    memcpy(cache_k, k + h * HEAD_SIZE, HEAD_SIZE * sizeof(float));
    memcpy(cache_v, v + h * HEAD_SIZE, HEAD_SIZE * sizeof(float));

    float *scores = state->att_scores; 
    
    /*
       --- SCORE CALCULATION ---
       For strict ablation, we could vectorize this Dot Product too.
       For this demo, we keep it manual/scalar loop to focus on the big kernels.
     */
    for (int t = 0; t <= pos; t++) {
      int past_offset = layer * (N_HEADS * MAX_SEQ_LEN * HEAD_SIZE) + 
                        h * (MAX_SEQ_LEN * HEAD_SIZE) + 
                        t * HEAD_SIZE;
      float *past_k = state->key_cache + past_offset;

      float score = 0.0f;
      for (int i = 0; i < HEAD_SIZE; i++) score += head_q[i] * past_k[i];
      score /= sqrtf((float)HEAD_SIZE);
      scores[t] = score;
    }

    softmax(scores, pos + 1);

    /* --- WEIGHTED SUM --- */
    float *head_out = state->attn_out + h * HEAD_SIZE;
    for (int i = 0; i < HEAD_SIZE; i++) head_out[i] = 0.0f;

    for (int t = 0; t <= pos; t++) {
      int past_offset = layer * (N_HEADS * MAX_SEQ_LEN * HEAD_SIZE) + 
                        h * (MAX_SEQ_LEN * HEAD_SIZE) + 
                        t * HEAD_SIZE;
      float *past_v = state->value_cache + past_offset;
      float prob = scores[t];
      for (int i = 0; i < HEAD_SIZE; i++) head_out[i] += prob * past_v[i];
    }
  }

  /* 3. Output Projection (Mapped) */
  matmul(out, state->attn_out, c_proj_w, c_proj_b, D_MODEL, D_MODEL);
}

void transformer_block(float *x, GPT2Weights *w, GPT2State *s, int layer, int pos) {
  memcpy(s->resid, x, D_MODEL * sizeof(float));

  /* LN 1 */
  layernorm(s->ln1_out, x, w->layers[layer].ln1_w, w->layers[layer].ln1_b, D_MODEL);

  /* Attention */
  float attn_out[D_MODEL];
  attention(s->attn_out, s->ln1_out, w->layers[layer].attn_w, w->layers[layer].attn_b, 
            w->layers[layer].attn_proj_w, w->layers[layer].attn_proj_b, s, layer, pos);
  
  /* Resid 1 */
  add(x, s->resid, s->attn_out, D_MODEL);
  memcpy(s->resid, x, D_MODEL * sizeof(float));

  /* LN 2 */
  layernorm(s->ln2_out, x, w->layers[layer].ln2_w, w->layers[layer].ln2_b, D_MODEL);

  /* MLP FC */
  matmul(s->mlp_hidden, s->ln2_out, w->layers[layer].mlp_fc_w, w->layers[layer].mlp_fc_b, D_MODEL, 4 * D_MODEL);
  
  /* GELU */
  gelu(s->mlp_hidden, 4 * D_MODEL);
  
  /* MLP Proj */
  float mlp_out[D_MODEL];
  matmul(s->mlp_out, s->mlp_hidden, w->layers[layer].mlp_proj_w, w->layers[layer].mlp_proj_b, 4 * D_MODEL, D_MODEL);

  /* Resid 2 */
  add(x, s->resid, s->mlp_out, D_MODEL);
}

void GPT2State_init(GPT2State *state) {
  long cache_size = (long)N_LAYERS * MAX_SEQ_LEN * D_MODEL; 
  state->key_cache = (float*)malloc(cache_size * sizeof(float));
  state->value_cache = (float*)malloc(cache_size * sizeof(float));
  state->x = (float *)malloc(D_MODEL * sizeof(float));
  state->logits = (float *)malloc(VOCAB_SIZE * sizeof(float));
  state->final = (float *)malloc(D_MODEL * sizeof(float));
  state->qkv = (float *)malloc(3 * D_MODEL * sizeof(float));
  state->att_scores = (float *)malloc(MAX_SEQ_LEN * sizeof(float));
  state->attn_out = (float *)malloc(D_MODEL * sizeof(float));
  state->ln1_out = (float *)malloc(D_MODEL * sizeof(float));
  state->ln2_out = (float *)malloc(D_MODEL * sizeof(float));
  state->mlp_hidden = (float *)malloc(4 * D_MODEL * sizeof(float));
  state->mlp_out = (float *)malloc(D_MODEL * sizeof(float));  
}

void GPT2State_free(GPT2State *state) {
  free(state->key_cache);
  free(state->value_cache);
  free(state->x);
  free(state->logits);
  free(state->final);
  free(state->qkv);
  free(state->att_scores);
  free(state->attn_out);
  free(state->ln1_out);
  free(state->ln2_out);
  free(state->mlp_hidden);
  free(state->mlp_out);
}

int main() {
  printf("ABLATIONS:\n");
  printf("MatMul:    %s\n", MODE_MATMUL);
  printf("LayerNorm: %s\n", MODE_LN);
  printf("Add:       %s\n", MODE_ADD);
  
  FILE *f = fopen("gpt2_weights.bin", "rb");
  if (!f) { printf("Error: gpt2_weights.bin not found\n"); return 1; }
  fseek(f, 0, SEEK_END);
  long filesize = ftell(f);
  fseek(f, 0, SEEK_SET);
  float *memory = (float*)malloc(filesize);
  if(!memory) { printf("Malloc failed\n"); return 1; }
  
  if (fread(memory, 1, filesize, f) != filesize) {
    printf("Error reading weights\n");
    free(memory); fclose(f); return 1;
  }
  fclose(f);

  GPT2Weights w;
  float *ptr = memory;
  w.wte = ptr; ptr += VOCAB_SIZE * D_MODEL;
  w.wpe = ptr; ptr += MAX_SEQ_LEN * D_MODEL;
  for (int i = 0; i < N_LAYERS; i++) {
    w.layers[i].ln1_w = ptr; ptr += D_MODEL;
    w.layers[i].ln1_b = ptr; ptr += D_MODEL;
    w.layers[i].attn_w = ptr; ptr += D_MODEL * 3 * D_MODEL;
    w.layers[i].attn_b = ptr; ptr += 3 * D_MODEL;
    w.layers[i].attn_proj_w = ptr; ptr += D_MODEL * D_MODEL;
    w.layers[i].attn_proj_b = ptr; ptr += D_MODEL;
    w.layers[i].ln2_w = ptr; ptr += D_MODEL;
    w.layers[i].ln2_b = ptr; ptr += D_MODEL;
    w.layers[i].mlp_fc_w = ptr; ptr += D_MODEL * 4 * D_MODEL;
    w.layers[i].mlp_fc_b = ptr; ptr += 4 * D_MODEL;
    w.layers[i].mlp_proj_w = ptr; ptr += 4 * D_MODEL * D_MODEL;
    w.layers[i].mlp_proj_b = ptr; ptr += D_MODEL;
  }
  w.ln_f_w = ptr; ptr += D_MODEL;
  w.ln_f_b = ptr; ptr += D_MODEL;
  w.lm_head = ptr;

  GPT2State state;
  GPT2State_init(&state);

  /* Prompt: "The quick brown fox jumps over the lazy" */
  int prompt_tokens[] = { 464, 2068, 7586, 21831, 18045, 625, 262, 16931 };
  int num_prompt = sizeof(prompt_tokens) / sizeof(int);
  int tokens_to_generate = 10; 

  printf("Prompt Length: %d. Generating %d tokens.\n", num_prompt, tokens_to_generate);

  int current_token = prompt_tokens[0];
  int pos = 0;
  
  clock_t start = clock();

  while (pos < num_prompt + tokens_to_generate) {
    /* Embedding */
    for(int i=0; i<D_MODEL; i++) {
      state.x[i] = w.wte[current_token * D_MODEL + i] + w.wpe[pos * D_MODEL + i];
    }

    /* Forward */
    for(int i=0; i<N_LAYERS; i++) {
      transformer_block(state.x, &w, &state, i, pos);
    }

    /* Final Norm */
    layernorm(state.final, state.x, w.ln_f_w, w.ln_f_b, D_MODEL);

    /* Next Token Logic */
    int next_token;
    if (pos < num_prompt - 1) {
      next_token = prompt_tokens[pos + 1];
    } else {
      /* Logits */
      matmul(state.logits, state.final, w.lm_head, NULL, D_MODEL, VOCAB_SIZE);
      
      float max_prob = -1e9;
      int argmax = 0;
      for(int i=0; i<VOCAB_SIZE; i++) {
        if (state.logits[i] > max_prob) {
          max_prob = state.logits[i];
          argmax = i;
        }
      }
      next_token = argmax;
      printf("Step %d | Token: %d\n", pos, next_token);
    }

    pos++;
    current_token = next_token;
    if (pos >= MAX_SEQ_LEN) break;
  }
  
  clock_t end = clock();
  double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
  printf("Inference finished in %f seconds.\n", time_spent);

  free(memory);
  GPT2State_free(&state);
  return 0;
}
