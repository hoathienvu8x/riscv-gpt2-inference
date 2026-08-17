#define _GNU_SOURCE
#define _ISOC99_SOURCE
#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

/*
 *  ABLATION CONFIGURATION
 *  1 = Use RISC-V Vector (RVV) Implementation
 *  0 = Use Scalar Baseline Implementation
 */

#define ENABLE_RVV_MATMUL     0 
#define ENABLE_RVV_LAYERNORM  0 
#define ENABLE_RVV_ADD        0
#define ENABLE_RVV_ACTIVATION 0

#if (ENABLE_RVV_MATMUL || ENABLE_RVV_LAYERNORM || ENABLE_RVV_ADD || ENABLE_RVV_ACTIVATION)
  #include <riscv_vector.h>
  #define HAS_RVV_HEADER 1
#endif

#define SQRT_2_PI 0.7978845608f
#define C_GELU 0.044715f

typedef struct {
  float *data;
  size_t pos, capacity;
} FloatBuffer;

typedef enum {
  GPT2_DTYPE_UNKNOWN = 0,
  GPT2_DTYPE_F32,
  GPT2_DTYPE_F16,
  GPT2_DTYPE_BF16,
  GPT2_DTYPE_I32,
  GPT2_DTYPE_I64
} GPT2DType;

typedef struct {
  GPT2DType dtype;
  int shape[4], ndim;
  uint64_t start_offset, end_offset;
  void *data;
} GPT2Tensor;

typedef enum {
  GPT2_ACTIVATION_UNKNOWN = 0,
  GPT2_ACTIVATION_GELU,
  GPT2_ACTIVATION_GELU_NEW,
  GPT2_ACTIVATION_GELU_FAST,
  GPT2_ACTIVATION_RELU,
  GPT2_ACTIVATION_SILU,
  GPT2_ACTIVATION_TANH
} GPT2Activation;

typedef struct {
  int vocab_size, n_positions, n_embd, n_layer, n_head;
  double layer_norm_epsilon;
  GPT2Activation activation_type;
} GPT2Config;

typedef struct {
  GPT2Tensor ln1_w, ln1_b;
  GPT2Tensor attn_w, attn_b;
  GPT2Tensor attn_proj_w, attn_proj_b;
  GPT2Tensor ln2_w, ln2_b;
  GPT2Tensor mlp_fc_w, mlp_fc_b;
  GPT2Tensor mlp_proj_w, mlp_proj_b;
} GPT2Layers;

typedef struct {
  GPT2Tensor wte; /* [VOCAB, D_MODEL] */
  GPT2Tensor wpe; /* [MAX_SEQ, D_MODEL] */
  GPT2Layers *layers;
  GPT2Tensor ln_f_w, ln_f_b;
  GPT2Tensor lm_head;
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
  FloatBuffer buf;
} GPT2State;


/* KERNELS */
void add(float *out, float *a, float *b, int size) {
  int i;
  #ifdef HAS_RVV_HEADER
  if (ENABLE_RVV_ADD) {
    size_t vl;
    for (i = 0; i < size; i += vl) {
      vl = __riscv_vsetvl_e32m8(size - i);
      vfloat32m8_t va = __riscv_vle32_v_f32m8(a + i, vl);
      vfloat32m8_t vb = __riscv_vle32_v_f32m8(b + i, vl);
      vfloat32m8_t vres = __riscv_vfadd_vv_f32m8(va, vb, vl);
      __riscv_vse32_v_f32m8(out + i, vres, vl);
    }
    return;
  }
  #endif
  for(i = 0; i < size; i++) {
    out[i] = a[i] + b[i];
  }
}

void apply_activate(float *x, int size, GPT2Activation act_type) {
  int i;
  #if defined(HAS_RVV_HEADER) && ENABLE_RVV_ACTIVATION
  size_t vl;
  for (i = 0; i < size; i += vl) {
    vl = __riscv_vsetvl_e32m8(size - i);
    {
      vfloat32m8_t xv = __riscv_vle32_v_f32m8(x + i, vl);
      float temp_buf[256];
      size_t j;
      vfloat32m8_t vres;

      if (vl > 256) vl = 256;
      __riscv_vse32_v_f32m8(temp_buf, xv, vl);

      for (j = 0; j < vl; j++) {
        float val = temp_buf[j];
        switch (act_type) {
          case GPT2_ACTIVATION_GELU: {
            float cube = C_GELU * val * val * val;
            float inner = SQRT_2_PI * (val + cube);
            temp_buf[j] = 0.5f * val * (1.0f + tanhf(inner));
            break;
          }
          case GPT2_ACTIVATION_GELU_NEW: {
            float inner = 0.7978845608f * (val + 0.044715f * val * val * val);
            temp_buf[j] = 0.5f * val * (1.0f + tanhf(inner));
            break;
          }
          case GPT2_ACTIVATION_GELU_FAST: {
            temp_buf[j] = val * (0.5f * (1.0f + tanhf(0.797885f * (val + 0.044715f * val * val * val))));
            break;
          }
          case GPT2_ACTIVATION_RELU: {
            temp_buf[j] = (val > 0.0f) ? val : 0.0f;
            break;
          }
          case GPT2_ACTIVATION_SILU: {
            temp_buf[j] = val / (1.0f + expf(-val));
            break;
          }
          case GPT2_ACTIVATION_TANH: {
            temp_buf[j] = tanhf(val);
            break;
          }
          case GPT2_ACTIVATION_UNKNOWN:
          default:
            break;
        }
      }

      vres = __riscv_vle32_v_f32m8(temp_buf, vl);
      __riscv_vse32_v_f32m8(x + i, vres, vl);
    }
  }
  return;
  #else

  /* Scalar Baseline Implementation */
  for(i = 0; i < size; i++) {
    float xv = x[i];
    switch (act_type) {
      case GPT2_ACTIVATION_GELU: {
        float cube = C_GELU * xv * xv * xv;
        float inner = SQRT_2_PI * (xv + cube);
        x[i] = 0.5f * xv * (1.0f + tanhf(inner));
        break;
      }
      case GPT2_ACTIVATION_GELU_NEW: {
        float inner = 0.7978845608f * (xv + 0.044715f * xv * xv * xv);
        x[i] = 0.5f * xv * (1.0f + tanhf(inner));
        break;
      }
      case GPT2_ACTIVATION_GELU_FAST: {
        x[i] = xv * (0.5f * (1.0f + tanhf(0.797885f * (xv + 0.044715f * xv * xv * xv))));
        break;
      }
      case GPT2_ACTIVATION_RELU: {
        x[i] = (xv > 0.0f) ? xv : 0.0f;
        break;
      }
      case GPT2_ACTIVATION_SILU: {
        x[i] = xv / (1.0f + expf(-xv));
        break;
      }
      case GPT2_ACTIVATION_TANH: {
        x[i] = tanhf(xv);
        break;
      }
      case GPT2_ACTIVATION_UNKNOWN:
      default:
        break;
    }
  }
  #endif
}

void layernorm(float *out, float *x, float *g, float *b, float eps, int size) {
  int i;
  float mean, var, inv_std;

  #if defined(HAS_RVV_HEADER) && ENABLE_RVV_LAYERNORM
  {
    size_t vl;
    vfloat32m1_t v_sum = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    int ptr = 0;
    vfloat32m1_t v_var;
    while(ptr < size) {
      vfloat32m8_t v_data;
      vl = __riscv_vsetvl_e32m8(size - ptr);
      v_data = __riscv_vle32_v_f32m8(x + ptr, vl);
      v_sum = __riscv_vfredusum_vs_f32m8_f32m1(v_data, v_sum, vl);
      ptr += vl;
    }
    mean = __riscv_vfmv_f_s_f32m1_f32(v_sum) / size;

    v_var = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    ptr = 0;
    while(ptr < size) {
      vfloat32m8_t v_data, v_diff, v_sq;
      vl = __riscv_vsetvl_e32m8(size - ptr);
      v_data = __riscv_vle32_v_f32m8(x + ptr, vl);
      v_diff = __riscv_vfsub_vf_f32m8(v_data, mean, vl);
      v_sq = __riscv_vfmul_vv_f32m8(v_diff, v_diff, vl);
      v_var = __riscv_vfredusum_vs_f32m8_f32m1(v_sq, v_var, vl);
      ptr += vl;
    }
    var = __riscv_vfmv_f_s_f32m1_f32(v_var) / size;
    inv_std = 1.0f / sqrtf(var + eps);

    ptr = 0;
    while(ptr < size) {
      vfloat32m8_t v_data, v_g, v_b, v_norm;
      vl = __riscv_vsetvl_e32m8(size - ptr);
      v_data = __riscv_vle32_v_f32m8(x + ptr, vl);
      v_g = __riscv_vle32_v_f32m8(g + ptr, vl);
      v_b = __riscv_vle32_v_f32m8(b + ptr, vl);
      
      v_norm = __riscv_vfsub_vf_f32m8(v_data, mean, vl);
      v_norm = __riscv_vfmul_vf_f32m8(v_norm, inv_std, vl);
      v_norm = __riscv_vfmacc_vv_f32m8(v_b, v_norm, v_g, vl);
      
      __riscv_vse32_v_f32m8(out + ptr, v_norm, vl);
      ptr += vl;
    }
  }
  #else
  mean = 0.0f;
  for(i = 0; i < size; i++) mean += x[i];
  mean /= size;
  
  var = 0.0f;
  for(i = 0; i < size; i++) {
    float diff = x[i] - mean;
    var += diff * diff;
  }
  var /= size;
  
  inv_std = 1.0f / sqrtf(var + eps);
  for(i = 0; i < size; i++) {
    out[i] = (x[i] - mean) * inv_std * g[i] + b[i];
  }
  #endif
}

void matmul(float *out, float *x, float *w, float *b, int dim_in, int dim_out) {
  int i, j;
  #if defined(HAS_RVV_HEADER) && ENABLE_RVV_MATMUL
  size_t vl;
  for (i = 0; i < dim_out; i += vl) {
    vfloat32m8_t v_acc;
    vl = __riscv_vsetvl_e32m8(dim_out - i);
    if (b != NULL) v_acc = __riscv_vle32_v_f32m8(b + i, vl);
    else v_acc = __riscv_vfmv_v_f_f32m8(0.0f, vl);

    for (j = 0; j < dim_in; j++) {
      float scalar_x = x[j];
      vfloat32m8_t v_w = __riscv_vle32_v_f32m8(w + (j * dim_out + i), vl);
      v_acc = __riscv_vfmacc_vf_f32m8(v_acc, scalar_x, v_w, vl);
    }
    __riscv_vse32_v_f32m8(out + i, v_acc, vl);
  }
  #else
  for (i = 0; i < dim_out; i++) {
    float val = (b != NULL) ? b[i] : 0.0f;
    for (j = 0; j < dim_in; j++) {
      val += x[j] * w[j * dim_out + i];
    }
    out[i] = val;
  }
  #endif
}

void softmax(float *x, int n) {
  int i;
  float max_val = x[0];
  float sum = 0.0f;
  float inv_sum;

  for (i = 1; i < n; i++) {
    if (x[i] > max_val) max_val = x[i];
  }
  for (i = 0; i < n; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  inv_sum = 1.0f / sum;
  for (i = 0; i < n; i++) {
    x[i] *= inv_sum;
  }
}

void attention(float *out, float *x, 
               float *c_attn_w, float *c_attn_b, 
               float *c_proj_w, float *c_proj_b,
               GPT2State *state, GPT2Config *config, int layer, int pos) {

  int n_embd = config->n_embd;
  int head_size = config->n_embd / config->n_head;
  float *q;
  float *k;
  float *v;
  int h;
  
  matmul(state->qkv, x, c_attn_w, c_attn_b, n_embd, 3 * n_embd);

  q = state->qkv;
  k = state->qkv + n_embd;
  v = state->qkv + 2 * n_embd;

  for (h = 0; h < config->n_head; h++) {
    float *head_q = q + h * head_size;
    int cache_offset = layer * (config->n_head * config->n_positions * head_size) + 
                       h * (config->n_positions * head_size) + 
                       pos * head_size;
    float *cache_k = state->key_cache + cache_offset;
    float *cache_v = state->value_cache + cache_offset;
    float *scores = state->att_scores; 
    float scale = 1.0f / sqrtf((float)head_size);
    float *head_out;
    int t;

    memcpy(cache_k, k + h * head_size, head_size * sizeof(float));
    memcpy(cache_v, v + h * head_size, head_size * sizeof(float));

    for (t = 0; t <= pos; t++) {
      int past_offset = layer * (config->n_head * config->n_positions * head_size) + 
                        h * (config->n_positions * head_size) + 
                        t * head_size;
      float *past_k = state->key_cache + past_offset;
      float score = 0.0f;
      int i;
      for (i = 0; i < head_size; i++) score += head_q[i] * past_k[i];
      scores[t] = score * scale;
    }

    softmax(scores, pos + 1);

    head_out = state->attn_out + h * head_size;
    memset(head_out, 0, head_size * sizeof(float));

    for (t = 0; t <= pos; t++) {
      int past_offset = layer * (config->n_head * config->n_positions * head_size) + 
                        h * (config->n_positions * head_size) + 
                        t * head_size;
      float *past_v = state->value_cache + past_offset;
      float prob = scores[t];
      int i;
      for (i = 0; i < head_size; i++) head_out[i] += prob * past_v[i];
    }
  }

  matmul(out, state->attn_out, c_proj_w, c_proj_b, n_embd, n_embd);
}

float fp16_to_fp32(uint16_t h) {
  uint32_t sign = ((h >> 15) & 1);
  uint32_t exp  = ((h >> 10) & 0x1f);
  uint32_t mant = (h & 0x3ff);
  
  uint32_t f_val;
  if (exp == 0) {
    if (mant == 0) {
      f_val = (sign << 31);
    } else {
      exp = 127 - 15 - 10;
      while (!(mant & 0x400)) {
        mant <<= 1;
        exp--;
      }
      mant &= 0x3ff;
      f_val = (sign << 31) | (exp << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    f_val = (sign << 31) | (0xff << 23) | (mant ? (mant << 13) : 0);
  } else {
    f_val = (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13);
  }
  float f;
  memcpy(&f, &f_val, sizeof(float));
  return f;
}

float bf16_to_fp32(uint16_t bf) {
  uint32_t f_val = ((uint32_t)bf) << 16;
  float f;
  memcpy(&f, &f_val, sizeof(float));
  return f;
}

int tensor_to_float(FloatBuffer *buf, const GPT2Tensor *tensor, float **out, int *size) {
  int total_elements = 1;
  int i;
  size_t required_pos;

  for (i = 0; i < tensor->ndim; i++) {
    total_elements *= tensor->shape[i];
  }
  if (tensor->ndim == 0) total_elements = 1;

  if (tensor->dtype == GPT2_DTYPE_F32) {
    *out = (float *)tensor->data;
    if (size) *size = total_elements;
    return 0;
  }

  required_pos = buf->pos + total_elements;
  if (required_pos > buf->capacity) {
    size_t new_cap = buf->capacity * 2;
    float *new_data;
    if (new_cap < required_pos) new_cap = required_pos + 1024;
    new_data = (float*)realloc(buf->data, new_cap * sizeof(float));
    if (!new_data) return -1;
    buf->data = new_data;
    buf->capacity = new_cap;
  }

  *out = buf->data + buf->pos;
  if (size) *size = total_elements;

  switch (tensor->dtype) {
    case GPT2_DTYPE_I32: {
      int i;
      int32_t *src = (int32_t *)tensor->data;
      for (i = 0; i < total_elements; i++) {
        (*out)[i] = (float)src[i];
      }
      break;
    }
    case GPT2_DTYPE_I64: {
      int i;
      int64_t *src = (int64_t *)tensor->data;
      for (i = 0; i < total_elements; i++) {
        (*out)[i] = (float)src[i];
      }
      break;
    }
    case GPT2_DTYPE_F16: {
      int i;
      uint16_t *src = (uint16_t *)tensor->data;
      for (i = 0; i < total_elements; i++) {
        (*out)[i] = fp16_to_fp32(src[i]);
      }
      break;
    }
    case GPT2_DTYPE_BF16: {
      int i;
      uint16_t *src = (uint16_t *)tensor->data;
      for (i = 0; i < total_elements; i++) {
        (*out)[i] = bf16_to_fp32(src[i]);
      }
      break;
    }
    default: return -1;
  }

  buf->pos = required_pos;
  return 0;
}

void transformer_block(float *x, GPT2Weights *w, GPT2State *s, GPT2Config *config, int layer, int pos) {
  int n_embd = config->n_embd;
  float *w_ptr1, *w_ptr2, *w_ptr3, *w_ptr4;
  memcpy(s->resid, x, n_embd * sizeof(float));

  tensor_to_float(&s->buf, &w->layers[layer].ln1_w, &w_ptr1, NULL);
  tensor_to_float(&s->buf, &w->layers[layer].ln1_b, &w_ptr2, NULL);

  layernorm(s->ln1_out, x, w_ptr1, w_ptr2, config->layer_norm_epsilon, n_embd);
  s->buf.pos = 0;

  tensor_to_float(&s->buf, &w->layers[layer].attn_w, &w_ptr1, NULL);
  tensor_to_float(&s->buf, &w->layers[layer].attn_b, &w_ptr2, NULL);
  tensor_to_float(&s->buf, &w->layers[layer].attn_proj_w, &w_ptr3, NULL);
  tensor_to_float(&s->buf, &w->layers[layer].attn_proj_b, &w_ptr4, NULL);

  attention(s->attn_out, s->ln1_out, w_ptr1, w_ptr2, 
            w_ptr3, w_ptr4, s, config, layer, pos);
  s->buf.pos = 0;
  
  add(x, s->resid, s->attn_out, n_embd);
  memcpy(s->resid, x, n_embd * sizeof(float));

  tensor_to_float(&s->buf, &w->layers[layer].ln2_w, &w_ptr1, NULL);
  tensor_to_float(&s->buf, &w->layers[layer].ln2_b, &w_ptr2, NULL);
  layernorm(s->ln2_out, x, w_ptr1, w_ptr2, config->layer_norm_epsilon, n_embd);
  s->buf.pos = 0;

  tensor_to_float(&s->buf, &w->layers[layer].mlp_fc_w, &w_ptr1, NULL);
  tensor_to_float(&s->buf, &w->layers[layer].mlp_fc_b, &w_ptr2, NULL);
  matmul(s->mlp_hidden, s->ln2_out, w_ptr1, w_ptr2, n_embd, 4 * n_embd);
  s->buf.pos = 0;
  
  apply_activate(s->mlp_hidden, 4 * n_embd, config->activation_type);
  tensor_to_float(&s->buf, &w->layers[layer].mlp_proj_w, &w_ptr1, NULL);
  tensor_to_float(&s->buf, &w->layers[layer].mlp_proj_b, &w_ptr2, NULL);
  matmul(s->mlp_out, s->mlp_hidden, w_ptr1, w_ptr2, 4 * n_embd, n_embd);

  add(x, s->resid, s->mlp_out, n_embd);
}

void GPT2Config_init(GPT2Config *config) {
  if (config) {
    config->n_layer = 12;
    config->n_embd = 768;
    config->n_head = 12;
    config->vocab_size = 50257;
    config->n_positions = 1024;
    config->layer_norm_epsilon = 1e-5f;
    config->activation_type = GPT2_ACTIVATION_GELU;
  }
}

void GPT2State_init(GPT2State *state, GPT2Config *config) {
  if (state && config) {
    long cache_size = (long)config->n_layer * config->n_positions * config->n_embd; 
    state->key_cache = (float*)malloc(cache_size * sizeof(float));
    state->value_cache = (float*)malloc(cache_size * sizeof(float));
    state->x = (float *)malloc(config->n_embd * sizeof(float));
    state->logits = (float *)malloc(config->vocab_size * sizeof(float));
    state->final = (float *)malloc(config->n_embd * sizeof(float));
    state->qkv = (float *)malloc(3 * config->n_embd * sizeof(float));
    state->att_scores = (float *)malloc(config->n_positions * sizeof(float));
    state->attn_out = (float *)malloc(config->n_embd * sizeof(float));
    state->ln1_out = (float *)malloc(config->n_embd * sizeof(float));
    state->ln2_out = (float *)malloc(config->n_embd * sizeof(float));
    state->mlp_hidden = (float *)malloc(4 * config->n_embd * sizeof(float));
    state->mlp_out = (float *)malloc(config->n_embd * sizeof(float));  
    state->buf.data = NULL;
    state->buf.pos = 0;
    state->buf.capacity = 0;

    if (
      !state->key_cache || !state->value_cache || !state->x || !state->logits || 
      !state->final || !state->qkv || !state->att_scores || !state->attn_out || 
      !state->ln1_out || !state->ln2_out || !state->mlp_hidden || !state->mlp_out
    ) {
      fprintf(stderr, "Error: Memory allocation failed in GPT2State_init\n");
      exit(EXIT_FAILURE);
    }
  } else {
    exit(EXIT_FAILURE);
  }
}

void GPT2State_free(GPT2State *state) {
  if (!state) return;
  if (state->key_cache) { free(state->key_cache); state->key_cache = NULL; }
  if (state->value_cache) { free(state->value_cache); state->value_cache = NULL; }
  if (state->x) { free(state->x); state->x = NULL; }
  if (state->logits) { free(state->logits); state->logits = NULL; }
  if (state->final) { free(state->final); state->final = NULL; }
  if (state->qkv) { free(state->qkv); state->qkv = NULL; }
  if (state->att_scores) { free(state->att_scores); state->att_scores = NULL; }
  if (state->attn_out) { free(state->attn_out); state->attn_out = NULL; }
  if (state->ln1_out) { free(state->ln1_out); state->ln1_out = NULL; }
  if (state->ln2_out) { free(state->ln2_out); state->ln2_out = NULL; }
  if (state->mlp_hidden) { free(state->mlp_hidden); state->mlp_hidden = NULL; }
  if (state->mlp_out) { free(state->mlp_out); state->mlp_out = NULL; }
  if (state->buf.data) {
    free(state->buf.data);
    state->buf.data = NULL;
    state->buf.pos = 0;
    state->buf.capacity = 0;
  }
}

typedef struct {
  float prob;
  int index;
} TokenProbability;

int compare_tokens(const void *a, const void *b) {
  const TokenProbability *x = (const TokenProbability *)a;
  const TokenProbability *y = (const TokenProbability *)b;
  if (x->prob > y->prob) return -1;
  if (x->prob < y->prob) return 1;
  return 0;
}

unsigned long long rng_seed = 0;
void set_seed(unsigned long long seed) {
  rng_seed = seed;
}
unsigned int random_u32(void) {
  rng_seed = rng_seed * 6364136223846793005ULL + 1442695040888963407ULL;
  return (unsigned int)(rng_seed >> 32);
}
float random_f32(void) {
  return (float)random_u32() / 4294967296.0f;
}

int sample(float *logits, int vocab_size, float temperature, int top_k, float top_p, TokenProbability *vocab_probs) {
  int i;
  float max_val;
  float sum;
  float inv_sum;
  int effective_vocab_size;
  float cumulative_sum;
  float r;
  float cdf;
  int next_token;

  if (temperature == 0.0f) {
    int max_i = 0;
    float max_p = logits[0];
    for (i = 1; i < vocab_size; i++) {
      if (logits[i] > max_p) {
        max_p = logits[i];
        max_i = i;
      }
    }
    return max_i;
  }

  for (i = 0; i < vocab_size; i++) {
    logits[i] /= temperature;
  }

  max_val = logits[0];
  for (i = 1; i < vocab_size; i++) {
    if (logits[i] > max_val) max_val = logits[i];
  }
  sum = 0.0f;
  for (i = 0; i < vocab_size; i++) {
    logits[i] = expf(logits[i] - max_val);
    sum += logits[i];
  }
  inv_sum = 1.0f / sum;
  for (i = 0; i < vocab_size; i++) {
    logits[i] *= inv_sum;
  }

  for (i = 0; i < vocab_size; i++) {
    vocab_probs[i].prob = logits[i];
    vocab_probs[i].index = i;
  }

  qsort(vocab_probs, vocab_size, sizeof(TokenProbability), compare_tokens);

  effective_vocab_size = vocab_size;
  if (top_k > 0 && top_k < vocab_size) {
    effective_vocab_size = top_k;
  }

  if (top_p > 0.0f && top_p < 1.0f) {
    float cumulative_prob = 0.0f;
    int last_idx = effective_vocab_size;
    int j;
    for (j = 0; j < effective_vocab_size; j++) {
      cumulative_prob += vocab_probs[j].prob;
      if (cumulative_prob > top_p) {
        last_idx = j + 1;
        break;
      }
    }
    effective_vocab_size = last_idx;
  }

  cumulative_sum = 0.0f;
  for (i = 0; i < effective_vocab_size; i++) {
    cumulative_sum += vocab_probs[i].prob;
  }
  
  r = random_f32() * cumulative_sum;
  cdf = 0.0f;
  next_token = vocab_probs[0].index;
  for (i = 0; i < effective_vocab_size; i++) {
    cdf += vocab_probs[i].prob;
    if (r < cdf) {
      next_token = vocab_probs[i].index;
      break;
    }
  }

  return next_token;
}

typedef struct {
  int max_gen_tokens;
  float temperature;
  int top_k;
  float top_p;
  unsigned long long seed;
} GPT2Param;

void generate(GPT2Weights *w, GPT2Config *config, GPT2Param *param,
  GPT2State *state, int *prompt_tokens, int num_prompt,
  TokenProbability *vocab_probs) {
  float *wte_ptr, *wpe_ptr, *ln_f_w_ptr, *ln_f_b_ptr, *lm_head_ptr;
  int current_token = prompt_tokens[0];
  int pos = 0;
  tensor_to_float(&state->buf, &w->wte, &wte_ptr, NULL);
  tensor_to_float(&state->buf, &w->wpe, &wpe_ptr, NULL);
  tensor_to_float(&state->buf, &w->ln_f_w, &ln_f_w_ptr, NULL);
  tensor_to_float(&state->buf, &w->ln_f_b, &ln_f_b_ptr, NULL);
  tensor_to_float(&state->buf, &w->lm_head, &lm_head_ptr, NULL);
  while (pos < num_prompt + param->max_gen_tokens) {
    int i;
    int next_token;

    for(i = 0; i < config->n_embd; i++) {
      state->x[i] = wte_ptr[current_token * config->n_embd + i] + wpe_ptr[pos * config->n_embd + i];
    }

    for(i = 0; i < config->n_layer; i++) {
      transformer_block(state->x, w, state, config, i, pos);
    }

    layernorm(state->final, state->x, ln_f_w_ptr, ln_f_b_ptr, config->layer_norm_epsilon, config->n_embd);

    if (pos < num_prompt - 1) {
      next_token = prompt_tokens[pos + 1];
    } else {
      matmul(state->logits, state->final, lm_head_ptr, NULL, config->n_embd, config->vocab_size);
      next_token = sample(state->logits, config->vocab_size, param->temperature, param->top_k, param->top_p, vocab_probs);
      printf("Step %d | Token: %d\n", pos, next_token);
    }

    pos++;
    current_token = next_token;
    if (pos >= config->n_positions) break;
  }
}

int main(void) {
  #if 0
  FILE *f;
  long filesize;
  float *memory;
  GPT2Param param;
  GPT2Weights w;
  GPT2Config config;
  float *ptr;
  GPT2State state;
  int prompt_tokens[8];
  int num_prompt;
  TokenProbability *vocab_probs;
  struct timespec start, end;
  double time_spent;
  int i;

  printf("ABLATIONS:\n");
  printf("MatMul:     %s\n", ENABLE_RVV_MATMUL ? "RVV" : "Scalar");
  printf("LayerNorm:  %s\n", ENABLE_RVV_LAYERNORM ? "RVV" : "Scalar");
  printf("Add:        %s\n", ENABLE_RVV_ADD ? "RVV" : "Scalar");
  printf("Activation: %s\n", ENABLE_RVV_ACTIVATION ? "RVV" : "Scalar");
  
  f = fopen("gpt2_weights.bin", "rb");
  if (!f) { printf("Error: gpt2_weights.bin not found\n"); return 1; }
  fseek(f, 0, SEEK_END);
  filesize = ftell(f);
  fseek(f, 0, SEEK_SET);
  memory = (float*)malloc(filesize);
  if(!memory) { printf("Malloc failed\n"); fclose(f); return 1; }
  
  if (fread(memory, 1, filesize, f) != (size_t)filesize) {
    printf("Error reading weights\n");
    free(memory); fclose(f); return 1;
  }
  fclose(f);

  param.max_gen_tokens = 10;
  param.temperature = 0.7f;
  param.top_k = 40;
  param.top_p = 0.9f;
  param.seed = 42;

  ptr = memory;
  GPT2Config_init(&config);
  
  w.wte = ptr; ptr += config.vocab_size * config.n_embd;
  w.wpe = ptr; ptr += config.n_positions * config.n_embd;
  w.layers = (GPT2Layers *)malloc(config.n_layer * sizeof(GPT2Layers));
  if (!w.layers) { printf("Malloc failed\n"); free(memory); return 1; }

  for (i = 0; i < config.n_layer; i++) {
    w.layers[i].ln1_w = ptr; ptr += config.n_embd;
    w.layers[i].ln1_b = ptr; ptr += config.n_embd;
    w.layers[i].attn_w = ptr; ptr += config.n_embd * 3 * config.n_embd;
    w.layers[i].attn_b = ptr; ptr += 3 * config.n_embd;
    w.layers[i].attn_proj_w = ptr; ptr += config.n_embd * config.n_embd;
    w.layers[i].attn_proj_b = ptr; ptr += config.n_embd;
    w.layers[i].ln2_w = ptr; ptr += config.n_embd;
    w.layers[i].ln2_b = ptr; ptr += config.n_embd;
    w.layers[i].mlp_fc_w = ptr; ptr += config.n_embd * 4 * config.n_embd;
    w.layers[i].mlp_fc_b = ptr; ptr += 4 * config.n_embd;
    w.layers[i].mlp_proj_w = ptr; ptr += 4 * config.n_embd * config.n_embd;
    w.layers[i].mlp_proj_b = ptr; ptr += config.n_embd;
  }
  w.ln_f_w = ptr; ptr += config.n_embd;
  w.ln_f_b = ptr; ptr += config.n_embd;
  w.lm_head = ptr;

  GPT2State_init(&state, &config);
  set_seed(param.seed);

  prompt_tokens[0] = 464;
  prompt_tokens[1] = 2068;
  prompt_tokens[2] = 7586;
  prompt_tokens[3] = 21831;
  prompt_tokens[4] = 18045;
  prompt_tokens[5] = 625;
  prompt_tokens[6] = 262;
  prompt_tokens[7] = 16931;

  num_prompt = sizeof(prompt_tokens) / sizeof(int);
  vocab_probs = (TokenProbability*)malloc(config.vocab_size * sizeof(TokenProbability));
  if (!vocab_probs) { printf("Malloc failed\n"); free(memory); free(w.layers); GPT2State_free(&state); return 1; }

  printf("Prompt Length: %d. Generating %d tokens.\n", num_prompt, param.max_gen_tokens);

  clock_gettime(CLOCK_MONOTONIC, &start);

  generate(&w, &config, &param, &state, prompt_tokens, num_prompt, vocab_probs);
  
  clock_gettime(CLOCK_MONOTONIC, &end);
  time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  printf("Inference finished in %f seconds.\n", time_spent);

  free(memory);
  free(w.layers);
  free(vocab_probs);
  GPT2State_free(&state);
  #endif
  return 0;
}
