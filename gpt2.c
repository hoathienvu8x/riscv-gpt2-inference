#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <ctype.h>
#include <stdint.h>
#include <limits.h>
#include <inttypes.h>
#include <math.h>

#ifndef PATH_MAX
  #define PATH_MAX 4096
#endif
#define MAX_INPUT_LEN 1024

#define INITIAL_CAPACITY 16
#define LOAD_FACTOR_THRESHOLD 0.7f

#define GPT2_OK 0
#define GPT2_ERR_NOMEM -1
#define GPT2_ERR_INVALID -2

struct gpt2_map_t {
  int fd;
  size_t size;
  unsigned char *ptr;
};

struct gpt2_string_t {
  char *buf;
  size_t len;
};

struct gpt2_json_t {
  const unsigned char *ptr;
  size_t pos, len;
};

enum gpt2_activation_t {
  gpt2_activation_unknown = 0,
  gpt2_activation_gelu,
  gpt2_activation_gelu_new,
  gpt2_activation_gelu_fast,
  gpt2_activation_relu,
  gpt2_activation_silu,
  gpt2_activation_tanh
};

struct gpt2_config_t {
  int vocab_size, n_positions, n_embd, n_layer, n_head;
  double layer_norm_epsilon;
  enum gpt2_activation_t activation_function;
  int scale_attn_weights, tie_word_embeddings;
  int bos_token_id, eos_token_id, pad_token_id, unk_token_id;
};

enum gpt2_cell_t {
  gpt2_cell_empty = 0,
  gpt2_cell_occupied,
  gpt2_cell_deleted
};

struct gpt2_node_t {
  struct gpt2_string_t key;
  int value;
  enum gpt2_cell_t state;
};

struct gpt2_hashmap_t {
  struct gpt2_node_t *buckets;
  size_t size, capacity;
};

struct gpt2_tokenize_t {
  struct gpt2_hashmap_t tokens, merges;
  unsigned int *id2token;
  char *byte_encoder[256];
  unsigned char byte_decoder[512];
};

enum gpt2_parse_t {
  gpt2_parse_object,
  gpt2_parse_add_token,
  gpt2_parse_token_item,
  gpt2_parse_vocab,
  gpt2_parse_merge,
  gpt2_parse_merge_item
};

struct gpt2_bpe_node_t {
  struct gpt2_string_t token;
  struct gpt2_bpe_node_t *next;
};

enum gpt2_dtype_t {
  gpt2_dtype_unknown = 0,
  gpt2_dtype_f32,
  gpt2_dtype_f16,
  gpt2_dtype_bf16,
  gpt2_dtype_i32,
  gpt2_dtype_i64
};

struct gpt2_tensor_t {
  enum gpt2_dtype_t dtype;
  int shape[4], ndim;
  uint64_t start_offset, end_offset;
  void *data;
};

struct gpt2_weight_t {
  struct gpt2_tensor_t ln_1_w, ln_1_b;
  struct gpt2_tensor_t qkv_w, qkv_b;
  struct gpt2_tensor_t attn_proj_w, attn_proj_b;
  struct gpt2_tensor_t ln_2_w, ln_2_b;
  struct gpt2_tensor_t fc_w, fc_b;
  struct gpt2_tensor_t proj_w, proj_b;
};

struct gpt2_float_t {
  float *data;
  size_t pos, capacity;
};

struct gpt2_state_t {
  float *x, *logits, *final;
  uint16_t *keys, *values;
  int seq_len;
  float *qkv, *att_scores, *attn_out;
  float *resid, *ln1, *ln2, *mlp_hidden, *mlp_out;
  struct gpt2_float_t buf;
};

struct gpt2_model_t {
  struct gpt2_config_t config;
  struct gpt2_tokenize_t tokenize;
  struct gpt2_map_t map;
  struct gpt2_tensor_t wte, wpe, lm_head;
  struct gpt2_tensor_t ln_f_w, ln_f_b;
  struct gpt2_weight_t *layers;
  struct gpt2_state_t state;
};

struct gpt2_param_t {
  int max_tokens, top_k;
  float temperature, top_p;
  unsigned int *seed;
};

struct gpt2_prob_t {
  float val;
  int idx;
};

/* String */
static struct gpt2_string_t gpt2_string_n(const char *buf, size_t len) {
  struct gpt2_string_t s;
  s.buf = (char *)buf, s.len = len;
  return s;
}

static struct gpt2_string_t gpt2_string_s(const char *buf) {
  return gpt2_string_n(buf, buf ? strlen(buf) : 0);
}

#define gpt2_string_t(s) gpt2_string_s(s)

static int gpt2_string_append(
  struct gpt2_string_t *s, const char *buf, size_t len
) {
  if (s != NULL && buf != NULL && len > 0) {
    size_t new_len = s->len + len;
    char *sc = (char *)calloc(1, new_len + 1);
    if (sc) {
      if (s->buf != NULL && s->len > 0) {
        memcpy(sc, s->buf, s->len);
        free(s->buf);
      }
      memcpy(sc + s->len, buf, len);
      sc[new_len] = '\0';
      s->buf = sc, s->len = new_len;
      return 0;
    }
  }
  return -1;
}

static int gpt2_string_cmp(
  const struct gpt2_string_t str1, const struct gpt2_string_t str2
) {
  size_t i = 0;
  while (i < str1.len && i < str2.len) {
    int c1 = str1.buf[i];
    int c2 = str2.buf[i];
    if (c1 < c2) return -1;
    if (c1 > c2) return 1;
    i++;
  }
  if (i < str1.len) return 1;
  if (i < str2.len) return -1;
  return 0;
}

static int gpt2_string_equal(
  const struct gpt2_string_t str1, const struct gpt2_string_t str2
) {
  return gpt2_string_cmp(str1, str2) == 0;
}

void gpt2_string_free(struct gpt2_string_t *s) {
  if (s) {
    if (s->buf) free(s->buf);
    s->buf = NULL, s->len = 0;
  }
}
/* String */

/* Map */
static int gpt2_map_init(const char *path, struct gpt2_map_t *map) {
  if (!map || !path || *path == '\0') return -1;
  map->fd = -1, map->ptr = NULL, map->size = 0;
  map->fd = open(path, O_RDONLY);
  if (map->fd != -1) {
    struct stat st;
    if (fstat(map->fd, &st) == -1 || st.st_size == 0) {
      close(map->fd), map->fd = -1;
      return -1;
    }
    map->size = (size_t)st.st_size;
    map->ptr = (unsigned char *)mmap(
      NULL, map->size, PROT_READ, MAP_SHARED, map->fd, 0
    );
    if (map->ptr != MAP_FAILED) {
      return 0;
    }
    close(map->fd), map->fd = -1, map->ptr = NULL;
  }
  return -1;
}

static void gpt2_map_free(struct gpt2_map_t *map) {
  if (!map) return;
  if (map->ptr && map->ptr != MAP_FAILED) {
    munmap(map->ptr, map->size);
    map->ptr = NULL;
  }
  if (map->fd >= 0) {
    close(map->fd), map->fd = -1;
  }
  map->size = 0;
}
/* Map */

/* JSON */
static int gpt2_json_next(struct gpt2_json_t *json) {
  if (!json || json->pos >= json->len) return EOF;
  return json->ptr[json->pos++];
}

static int gpt2_json_peek(struct gpt2_json_t *json) {
  if (!json || json->pos >= json->len) return EOF;
  return json->ptr[json->pos];
}

static int gpt2_json_skip(struct gpt2_json_t *json) {
  if (!json) return EOF;
  while (json->pos < json->len) {
    if (!isspace((unsigned char)json->ptr[json->pos])) break;
    json->pos++;
  }
  return (json->pos < json->len ? json->ptr[json->pos] : EOF);
}

static int gpt2_json_get_null(struct gpt2_json_t *json) {
  if (json) {
    int c = gpt2_json_skip(json);
    if (c == 'n') {
      const char *s = "null";
      int i;
      for (i = 0; i < 4; i++) {
        if (gpt2_json_next(json) != s[i]) return -1;
      }
      c = gpt2_json_peek(json);
      if (c != EOF && isalnum((unsigned char)c)) return -1;
      return 0;
    }
  }
  return -1;
}

static int gpt2_json_get_boolean(struct gpt2_json_t *json, int *bval) {
  if (json && bval) {
    int i, c = gpt2_json_skip(json);
    if (c == 'n') {
      if (gpt2_json_get_null(json)) return -1;
      *bval = 0;
      return 1;
    }
    if (c == 't') {
      const char *t = "true";
      for (i = 0; i < 4; i++) {
        if (gpt2_json_next(json) != t[i]) return -1;
      }
      c = gpt2_json_peek(json);
      if (c != EOF && isalnum((unsigned char)c)) return -1;
      *bval = 1;
    } else if (c == 'f') {
      const char *f = "false";
      for (i = 0; i < 5; i++) {
        if (gpt2_json_next(json) != f[i]) return 0;
      }
      c = gpt2_json_peek(json);
      if (c != EOF && isalnum((unsigned char)c)) return -1;
      *bval = 0;
    } else {
      return -1;
    }
    return 0;
  }
  return -1;
}

static int gpt2_parse_uint64(const char *buf, size_t len, uint64_t *val) {
  if (len > 0) {
    uint64_t res = 0;
    size_t i;
    for (i = 0; i < len; i++) {
      char c = buf[i];
      if (!isdigit((unsigned char)c)) return -1;
      int digit = c - '0';
      if (res > (UINT64_MAX - digit) / 10) return -1;
      res = res * 10 + digit;
    }
    *val = res;
    return 0;
  }
  return -1;
}
static int gpt2_parse_double(const char *buf, size_t len, double *val) {
  if (len > 0) {
    size_t i = 0;
    double int_part = 0.0, frac_part = 0.0, divisor = 1.0, result = 0;
    int negative = 0, has_digits = 0;
    if (buf[i] == '-') {
      negative = 1, i++;
    } else if (buf[i] == '+') {
      i++;
    }
    if (i >= len) return -1;
    while (i < len && isdigit((unsigned char)buf[i])) {
      int_part = int_part * 10.0 + (buf[i] - '0');
      has_digits = 1, i++;
    }
    if (i < len && buf[i] == '.') {
      i++;
      while (i < len && isdigit((unsigned char)buf[i])) {
        frac_part = frac_part * 10.0 + (buf[i] - '0');
        divisor *= 10.0, has_digits = 1, i++;
      }
    }
    if (!has_digits) return -1;
    result = int_part + (frac_part / divisor);
    if (i < len && (buf[i] == 'e' || buf[i] == 'E')) {
      double mult = 1.0;
      int exp_negative = 0, exponent = 0;
      if (i++ >= len) return -1;
      if (buf[i] == '-') {
        exp_negative = 1, i++;
      } else if (buf[i] == '+') {
        i++;
      }
      if (i >= len || !isdigit((unsigned char)buf[i])) return -1;
      while (i < len && isdigit((unsigned char)buf[i])) {
        exponent = exponent * 10 + (buf[i] - '0');
        i++;
      }
      for (int e = 0; e < exponent; e++) {
        mult *= 10.0;
      }
      if (exp_negative) {
        result /= mult;
      } else {
        result *= mult;
      }
    }
    if (i < len) return -1;
    *val = negative ? -result : result;
    return 0;
  }
  return -1;
}

static int gpt2_json_get_uint64(struct gpt2_json_t *json, uint64_t *val) {
  if (json && val) {
    char num_buf[64];
    size_t idx = 0;
    int c = gpt2_json_skip(json);
    if (c == 'n') {
      if (gpt2_json_get_null(json)) return -1;
      *val = 0;
      return 0;
    }
    while (idx < sizeof(num_buf) - 1) {
      c = gpt2_json_peek(json);
      if (c == EOF) break;
      if (isdigit((unsigned char)c) || c == '+' || c == 'e' || c == 'E') {
        num_buf[idx++] = (char)gpt2_json_next(json);
      } else {
        break;
      }
    }
    if (idx == 0) return -1;
    return gpt2_parse_uint64(num_buf, idx, val);
  }
  return -1;
}

static int gpt2_json_get_number(struct gpt2_json_t *json, double *val) {
  if (json && val) {
    char num_buf[64];
    size_t idx = 0;
    int c = gpt2_json_skip(json);
    if (c == 'n') {
      if (gpt2_json_get_null(json)) return -1;
      *val = 0.0;
      return 0;
    }
    if (c == 't' || c == 'f') {
      int bval = 0;
      if (gpt2_json_get_boolean(json, &bval)) return -1;
      *val = bval ? 1.0 : 0.0;
      return 0;
    }
    while (idx < sizeof(num_buf) - 1) {
      c = gpt2_json_peek(json);
      if (c == EOF) break;
      if (
        isdigit((unsigned char)c) || c == '-' || c == '+' ||
        c == '.' || c == 'e' || c == 'E'
      ) {
        num_buf[idx++] = (char)gpt2_json_next(json);
      } else {
        break;
      }
    }
    if (idx == 0) return -1;
    return gpt2_parse_double(num_buf, idx, val);
  }
  return -1;
}

static int gpt2_json_skip_value(struct gpt2_json_t *json) {
  if (json) {
    int c = gpt2_json_skip(json);
    if (c == EOF) return -1;
    if (c == '"') {
      gpt2_json_next(json);
      int escaped = 0;
      while (1) {
        c = gpt2_json_next(json);
        if (c == EOF) return -1;
        if (escaped) {
          escaped = 0;
        } else if (c == '\\') {
          escaped = 1;
        } else if (c == '"') {
          break;
        }
      }
    } else if (c == '{' || c == '[') {
      int depth = 1;
      char open_char = (char)c;
      char close_char = (c == '{') ? '}' : ']';
      gpt2_json_next(json);
      while (depth > 0) {
        c = gpt2_json_next(json);
        if (c == EOF) return 0;
        if (c == '"') {
          int escaped = 0;
          while (1) {
            c = gpt2_json_next(json);
            if (c == EOF) return 0;
            if (escaped) {
              escaped = 0;
            } else if (c == '\\') {
              escaped = 1;
            } else if (c == '"') {
              break;
            }
          }
        } else if (c == open_char) {
          depth++;
        } else if (c == close_char) {
          depth--;
        }
      }
    } else {
      while (1) {
        c = gpt2_json_peek(json);
        if (
          c == EOF || isspace((unsigned char)c) ||
          c == ',' || c == '}' || c == ']'
        ) {
          break;
        }
        gpt2_json_next(json);
      }
    }
    return 0;
  }
  return -1;
}

static int gpt2_json_get_raw_string(struct gpt2_json_t *json, struct gpt2_string_t *str) {
  if (json && str) {
    size_t len = 0, pos;
    int c;
    str->buf = NULL, str->len = 0;
    if (gpt2_json_skip(json) != '"') return -1;
    gpt2_json_next(json);
    pos = json->pos;
    while ((c = gpt2_json_next(json)) != EOF) {
      if (c == '\\') {
        len++, gpt2_json_next(json);
      } else if (c == '"') {
        str->buf = (char *)json->ptr + pos, str->len = len;
        c = gpt2_json_skip(json);
        if (
          c != EOF && !isspace(c) && c != ',' &&
          c != '}' && c != ']' && c != ':'
        ) {
          str->buf = NULL, str->len = 0;
          return -1;
        }
        return 0;
      }
      len++;
    }
  }
  return -1;
}

static int gpt2_json_get_string(
  struct gpt2_json_t *json, struct gpt2_string_t *str
) {
  if (json && str) {
    char stack_buf[MAX_INPUT_LEN];
    int escaped = 0;
    size_t stack_idx = 0, in_idx = 0;

    str->buf = NULL, str->len = 0;
    struct gpt2_string_t raw_str;
    if (gpt2_json_get_raw_string(json, &raw_str) != 0) {
      return -1;
    }

    while (in_idx < raw_str.len) {
      char c = raw_str.buf[in_idx++];
      if (stack_idx + 4 >= sizeof(stack_buf)) {
        if (stack_idx > 0) {
          if (gpt2_string_append(str, stack_buf, stack_idx) != 0) {
            if (str->buf) free(str->buf);
            str->buf = NULL, str->len = 0;
            return -1;
          }
          stack_idx = 0;
        }
      }

      if (escaped) {
        char decoded_char = c;
        int is_unicode = 0;
        unsigned int codepoint = 0;

        switch (c) {
          case 'n': decoded_char = '\n'; break;
          case 't': decoded_char = '\t'; break;
          case 'r': decoded_char = '\r'; break;
          case 'b': decoded_char = '\b'; break;
          case 'f': decoded_char = '\f'; break;
          case '"': decoded_char = '"';  break;
          case '\\': decoded_char = '\\'; break;
          case '/': decoded_char = '/';  break;
          case 'u': {
            if (in_idx + 4 <= raw_str.len) {
              unsigned int high_surrogate = 0;
              int i, valid_hex = 1;
              for (i = 0; i < 4; i++) {
                char h = raw_str.buf[in_idx++];
                high_surrogate <<= 4;
                if (h >= '0' && h <= '9') {
                  high_surrogate += (h - '0');
                } else if (h >= 'a' && h <= 'f') {
                  high_surrogate += (h - 'a' + 10);
                } else if (h >= 'A' && h <= 'F') {
                  high_surrogate += (h - 'A' + 10);
                } else {
                  valid_hex = 0;
                  break;
                }
              }

              if (valid_hex) {
                if (high_surrogate >= 0xD800 && high_surrogate <= 0xDBFF) {
                  if (
                    in_idx + 6 <= raw_str.len &&
                    raw_str.buf[in_idx] == '\\' &&
                    raw_str.buf[in_idx + 1] == 'u'
                  ) {
                    size_t temp_idx = in_idx + 2;
                    unsigned int low_surrogate = 0;
                    int low_valid = 1;
                    for (i = 0; i < 4; i++) {
                      char h = raw_str.buf[temp_idx++];
                      low_surrogate <<= 4;
                      if (h >= '0' && h <= '9') {
                        low_surrogate += (h - '0');
                      } else if (h >= 'a' && h <= 'f') {
                        low_surrogate += (h - 'a' + 10);
                      } else if (h >= 'A' && h <= 'F') {
                        low_surrogate += (h - 'A' + 10);
                      } else {
                        low_valid = 0;
                        break;
                      }
                    }
                    if (low_valid && low_surrogate >= 0xDC00 && low_surrogate <= 0xDFFF) {
                      codepoint = 0x10000 + (((high_surrogate - 0xD800) << 10) | (low_surrogate - 0xDC00));
                      in_idx = temp_idx;
                      is_unicode = 1;
                    }
                  }
                  if (!is_unicode) {
                    codepoint = high_surrogate;
                    is_unicode = 1;
                  }
                } else {
                  codepoint = high_surrogate;
                  is_unicode = 1;
                }
              } else {
                decoded_char = '?';
              }
            } else {
              in_idx = raw_str.len;
              decoded_char = '?';
            }
            break;
          }
          default: decoded_char = c; break;
        }

        if (is_unicode) {
          if (codepoint <= 0x7F) {
            stack_buf[stack_idx++] = (char)codepoint;
          } else if (codepoint <= 0x7FF) {
            stack_buf[stack_idx++] = (char)(0xC0 | (codepoint >> 6));
            stack_buf[stack_idx++] = (char)(0x80 | (codepoint & 0x3F));
          } else if (codepoint <= 0xFFFF) {
            stack_buf[stack_idx++] = (char)(0xE0 | (codepoint >> 12));
            stack_buf[stack_idx++] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
            stack_buf[stack_idx++] = (char)(0x80 | (codepoint & 0x3F));
          } else {
            stack_buf[stack_idx++] = (char)(0xF0 | (codepoint >> 18));
            stack_buf[stack_idx++] = (char)(0x80 | ((codepoint >> 12) & 0x3F));
            stack_buf[stack_idx++] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
            stack_buf[stack_idx++] = (char)(0x80 | (codepoint & 0x3F));
          }
        } else {
          stack_buf[stack_idx++] = decoded_char;
        }
        escaped = 0;
      } else if (c == '\\') {
        escaped = 1;
      } else {
        stack_buf[stack_idx++] = c;
      }
    }
    if (stack_idx > 0) {
      if (gpt2_string_append(str, stack_buf, stack_idx) != 0) {
        if (str->buf) free(str->buf);
        str->buf = NULL, str->len = 0;
        return -1;
      }
    }
    if (!str->buf) {
      str->buf = (char *)calloc(1, 1);
      str->len = 0;
    }
    return 0;
  }
  return -1;
}
/* JSON */

static int gpt2_load_config(const char *path, struct gpt2_config_t *config) {
  struct gpt2_map_t map = {0};
  struct gpt2_json_t json = {0};
  int c, ok = GPT2_OK;
  uint64_t val = 0;
  if (gpt2_map_init(path, &map)) {
    return GPT2_ERR_INVALID;
  }
  json.ptr = map.ptr, json.pos = 0, json.len = map.size;
  c = gpt2_json_skip(&json);
  if (c != '{') {
    gpt2_map_free(&map);
    return GPT2_ERR_INVALID;
  }
  gpt2_json_next(&json);
  config->bos_token_id = -1;
  config->eos_token_id = -1;
  config->pad_token_id = -1;
  config->unk_token_id = -1;
  while (1) {
    c = gpt2_json_skip(&json);
    if (c == '}' || c == EOF) {
      gpt2_json_next(&json);
      break;
    }
    struct gpt2_string_t key = {NULL, 0};
    if (gpt2_json_get_raw_string(&json, &key)) {
      ok = GPT2_ERR_INVALID;
      break;
    }
    c = gpt2_json_skip(&json);
    if (c != ':') {
      ok = GPT2_ERR_INVALID;
      break;
    }
    gpt2_json_next(&json);
    if (gpt2_string_equal(key, gpt2_string_t("vocab_size"))) {
      if (gpt2_json_get_uint64(&json, &val)) {
        ok = GPT2_ERR_INVALID;
        break;
      } else {
        config->vocab_size = (int)val;
      }
    } else if (gpt2_string_equal(key, gpt2_string_t("n_positions"))) {
      if (gpt2_json_get_uint64(&json, &val)) {
        ok = GPT2_ERR_INVALID;
        break;
      } else {
        config->n_positions = (int)val;
      }
    } else if (gpt2_string_equal(key, gpt2_string_t("n_embd"))) {
      if (gpt2_json_get_uint64(&json, &val)) {
        ok = GPT2_ERR_INVALID;
        break;
      } else {
        config->n_embd = (int)val;
      }
    } else if (gpt2_string_equal(key, gpt2_string_t("n_layer"))) {
      if (gpt2_json_get_uint64(&json, &val)) {
        ok = GPT2_ERR_INVALID;
        break;
      } else {
        config->n_layer = (int)val;
      }
    } else if (gpt2_string_equal(key, gpt2_string_t("n_head"))) {
      if (gpt2_json_get_uint64(&json, &val)) {
        ok = GPT2_ERR_INVALID;
        break;
      } else {
        config->n_head = (int)val;
      }
    } else if (gpt2_string_equal(key, gpt2_string_t("layer_norm_epsilon"))) {
      if (gpt2_json_get_number(&json, &config->layer_norm_epsilon)) {
        config->layer_norm_epsilon = 1e-5;
      }
    } else if (gpt2_string_equal(key, gpt2_string_t("activation_function"))) {
      config->activation_function = gpt2_activation_gelu;
      if (gpt2_json_get_string(&json, &key) == 0) {
        if (gpt2_string_cmp(key, gpt2_string_t("gelu_new")) == 0) {
          config->activation_function = gpt2_activation_gelu_new;
        } else if (gpt2_string_cmp(key, gpt2_string_t("gelu_fast")) == 0) {
          config->activation_function = gpt2_activation_gelu_fast;
        } else if (gpt2_string_cmp(key, gpt2_string_t("relu")) == 0) {
          config->activation_function = gpt2_activation_relu;
        } else if (gpt2_string_cmp(key, gpt2_string_t("silu")) == 0) {
          config->activation_function = gpt2_activation_silu;
        } else if (gpt2_string_cmp(key, gpt2_string_t("tanh")) == 0) {
          config->activation_function = gpt2_activation_tanh;
        }
        gpt2_string_free(&key);
      }
    } else if (gpt2_string_equal(key, gpt2_string_t("scale_attn_weights"))) {
      if (gpt2_json_get_boolean(&json, &config->scale_attn_weights)) {
        config->scale_attn_weights = 0;
      }
    } else if (gpt2_string_equal(key, gpt2_string_t("tie_word_embeddings"))) {
      if (gpt2_json_get_boolean(&json, &config->tie_word_embeddings)) {
        config->tie_word_embeddings = 0;
      }
    } else if (gpt2_string_equal(key, gpt2_string_t("bos_token_id"))) {
      if (gpt2_json_get_uint64(&json, &val) == 0) {
        config->bos_token_id = (int)val;
      }
    } else if (gpt2_string_equal(key, gpt2_string_t("eos_token_id"))) {
      if (gpt2_json_get_uint64(&json, &val) == 0) {
        config->eos_token_id = (int)val;
      }
    } else if (gpt2_string_equal(key, gpt2_string_t("pad_token_id"))) {
      if (gpt2_json_get_uint64(&json, &val) == 0) {
        config->pad_token_id = (int)val;
      }
    } else if (gpt2_string_equal(key, gpt2_string_t("unk_token_id"))) {
      if (gpt2_json_get_uint64(&json, &val) == 0) {
        config->unk_token_id = (int)val;
      }
    } else {
      if (gpt2_json_skip_value(&json)) {
        ok = GPT2_ERR_INVALID;
        break;
      }
    }
    c = gpt2_json_skip(&json);
    if (c == ',') {
      gpt2_json_next(&json);
    } else if (c != '}') {
      ok = GPT2_OK;
      break;
    }
  }
  gpt2_map_free(&map);
  return ok;
}

/* Hashmap */
static unsigned long gpt2_hash_bytes(const char *buf, size_t len) {
  unsigned long hash = 5381;
  size_t i;
  for (i = 0; i < len; i++) {
    hash = ((hash << 5) + hash) + (unsigned char)buf[i];
  }
  return hash;
}

static int gpt2_hashmap_insert(
  struct gpt2_hashmap_t *map, const struct gpt2_string_t *key,
  int value, int copy_key
) {
  unsigned long hash;
  size_t index, original_index;
  if (!map || !key || !key->buf) return GPT2_ERR_INVALID;

  hash = gpt2_hash_bytes(key->buf, key->len);
  index = hash % map->capacity;
  original_index = index;

  while (map->buckets[index].state == gpt2_cell_occupied) {
    if (gpt2_string_equal(map->buckets[index].key, *key)) {
      map->buckets[index].value = value;
      return GPT2_OK;
    }
    index = (index + 1) % map->capacity;
    if (index == original_index) {
      return GPT2_ERR_NOMEM;
    }
  }

  if (map->buckets[index].state == gpt2_cell_empty) {
    map->size++;
  } else {
    gpt2_string_free(&(map->buckets[index].key));
  }

  if (copy_key) {
    char *new_buf = (char *)malloc(key->len);
    if (!new_buf) return GPT2_ERR_NOMEM;
    memcpy(new_buf, key->buf, key->len);

    map->buckets[index].key.buf = new_buf;
    map->buckets[index].key.len = key->len;
  } else {
    map->buckets[index].key.buf = key->buf;
    map->buckets[index].key.len = key->len;
  }

  map->buckets[index].value = value;
  map->buckets[index].state = gpt2_cell_occupied;

  return GPT2_OK;
}

static int gpt2_hashmap_init(struct gpt2_hashmap_t *map, size_t capacity) {
  if (map && capacity > 0) {
    map->capacity = capacity;
    map->size = 0;
    map->buckets = (struct gpt2_node_t *)calloc(
      map->capacity, sizeof(struct gpt2_node_t)
    );
    if (!map->buckets) {
      return GPT2_ERR_NOMEM;
    }
    return GPT2_OK;
  }
  return GPT2_ERR_INVALID;
}

static void gpt2_hashmap_free(struct gpt2_hashmap_t *map) {
  if (map && map->buckets) {
    size_t i;
    for (i = 0; i < map->capacity; i++) {
      if (map->buckets[i].state == gpt2_cell_occupied) {
        gpt2_string_free(&(map->buckets[i].key));
      }
    }
    free(map->buckets);
    map->size = 0, map->capacity = 0;
  }
}

static int gpt2_hashmap_rehash(struct gpt2_hashmap_t *map) {
  size_t old_capacity, new_capacity, i;
  struct gpt2_node_t *old_buckets = NULL, *new_buckets = NULL;
  struct gpt2_hashmap_t temp_map;

  if (!map) return GPT2_ERR_INVALID;

  old_capacity = map->capacity;
  old_buckets = map->buckets;

  new_capacity = old_capacity * 2;
  new_buckets = (struct gpt2_node_t *)calloc(
    new_capacity, sizeof(struct gpt2_node_t)
  );
  if (!new_buckets) {
    return GPT2_ERR_NOMEM;
  }

  temp_map.capacity = new_capacity;
  temp_map.size = 0;
  temp_map.buckets = new_buckets;

  for (i = 0; i < old_capacity; i++) {
    if (old_buckets[i].state == gpt2_cell_occupied) {
      int err = gpt2_hashmap_insert(
        &temp_map, &(old_buckets[i].key), old_buckets[i].value, 0
      );
      if (err != GPT2_OK) {
        free(new_buckets);
        return err;
      }
    }
  }
  free(old_buckets);

  map->capacity = temp_map.capacity;
  map->size = temp_map.size;
  map->buckets = temp_map.buckets;

  return GPT2_OK;
}

static int gpt2_hashmap_put(
  struct gpt2_hashmap_t *map, const struct gpt2_string_t *key, int value
) {
  if (!map || !key) return GPT2_ERR_INVALID;

  if ((float)(map->size + 1) / map->capacity > LOAD_FACTOR_THRESHOLD) {
    int err = gpt2_hashmap_rehash(map);
    if (err != GPT2_OK) return err;
  }
  return gpt2_hashmap_insert(map, key, value, 0);
}

static int* gpt2_hashmap_get(
  const struct gpt2_hashmap_t *map, const struct gpt2_string_t *key
) {
  unsigned long hash;
  size_t index, original_index;
  if (!map || !key || !key->buf) return NULL;

  hash = gpt2_hash_bytes(key->buf, key->len);
  index = hash % map->capacity;
  original_index = index;

  while (map->buckets[index].state != gpt2_cell_empty) {
    if (
      map->buckets[index].state == gpt2_cell_occupied &&
      gpt2_string_equal(map->buckets[index].key, *key)
    ) {
      return &(map->buckets[index].value);
    }
    index = (index + 1) % map->capacity;
    if (index == original_index) break;
  }
  return NULL;
}

static int* gpt2_hashmap_get_merge(
  struct gpt2_hashmap_t *map,
  const struct gpt2_string_t *token1, const struct gpt2_string_t *token2
) {
  struct gpt2_string_t merge_key = {0};
  int *val = NULL;
  if (!map || !token1 || !token2) return NULL;

  merge_key.len = token1->len + token2->len;
  merge_key.buf = (char *)malloc(merge_key.len);
  if (!merge_key.buf) return NULL;

  memcpy(merge_key.buf, token1->buf, token1->len);
  memcpy(merge_key.buf + token1->len, token2->buf, token2->len);

  val = gpt2_hashmap_get(map, &merge_key);

  gpt2_string_free(&merge_key);
  return val;
}

static int gpt2_get_token(
  const struct gpt2_tokenize_t *tokenizer,
  int id, struct gpt2_string_t *token
) {
  if (
    !tokenizer || id < 0 ||
    (size_t)id >= tokenizer->tokens.size ||
    !tokenizer->id2token
  ) {
    return -1;
  }
  token->buf = NULL, token->len = 0;
  unsigned int index = tokenizer->id2token[id];
  if (tokenizer->tokens.buckets[index].state == gpt2_cell_occupied) {
    token->buf = tokenizer->tokens.buckets[index].key.buf;
    token->len = tokenizer->tokens.buckets[index].key.len;
    return 0;
  }
  return -1;
}

static void gpt2_init_byte_maps(struct gpt2_tokenize_t *tokenizer) {
  if (!tokenizer) return;
  int bs[256], cs[256], n = 0;
  for (int b = 0x21; b <= 0x7E; b++) { bs[n] = b; cs[n] = b; n++; }
  for (int b = 0xA1; b <= 0xAC; b++) { bs[n] = b; cs[n] = b; n++; }
  for (int b = 0xAE; b <= 0xFF; b++) { bs[n] = b; cs[n] = b; n++; }

  int n_val = 0;
  for (int b = 0; b < 256; b++) {
    int found = 0;
    for (int i = 0; i < n; i++) {
      if (bs[i] == b) { found = 1; break; }
    }
    if (!found) {
      bs[n + n_val] = b;
      cs[n + n_val] = 256 + n_val;
      n_val++;
    }
  }

  for (int i = 0; i < 256; i++) {
    tokenizer->byte_encoder[i] = NULL;
  }
  for (int i = 0; i < 512; i++) {
    tokenizer->byte_decoder[i] = 0;
  }

  for (int i = 0; i < 256; i++) {
    int b = bs[i];
    int c = cs[i];
    char *utf8_str = (char *)malloc(4);

    if (c < 0x80) {
      utf8_str[0] = (char)c;
      utf8_str[1] = '\0';
    } else if (c < 0x800) {
      utf8_str[0] = (char)(0xC0 | (c >> 6));
      utf8_str[1] = (char)(0x80 | (c & 0x3F));
      utf8_str[2] = '\0';
    } else {
      utf8_str[0] = (char)(0xE0 | (c >> 12));
      utf8_str[1] = (char)(0x80 | ((c >> 6) & 0x3F));
      utf8_str[2] = (char)(0x80 | (c & 0x3F));
      utf8_str[3] = '\0';
    }

    tokenizer->byte_encoder[b] = utf8_str;
    if (c < 512) {
      tokenizer->byte_decoder[c] = (unsigned char)b;
    }
  }
}
/* Hashmap */

static int gpt2_load_tokenize(const char *path, struct gpt2_tokenize_t *tokenizer, struct gpt2_config_t *config) {
  struct gpt2_map_t map = {0};
  struct gpt2_json_t json = {0};
  struct gpt2_string_t key = {0};
  uint64_t current_id = (uint64_t)-1, rank = 0;
  struct gpt2_string_t sval = {0};
  int ok = 0, c, state = gpt2_parse_object;
  size_t n_capacity = INITIAL_CAPACITY;
  if (config) {
    if ((size_t)config->vocab_size > n_capacity) {
      n_capacity = config->vocab_size;
    }
  }

  if (!path || *path == '\0' || !tokenizer) {
    return GPT2_ERR_INVALID;
  }

  if (gpt2_map_init(path, &map)) {
    return GPT2_ERR_INVALID;
  }
  if (gpt2_hashmap_init(&tokenizer->tokens, n_capacity) != GPT2_OK) {
    gpt2_map_free(&map);
    return GPT2_ERR_INVALID;
  }
  if (gpt2_hashmap_init(&tokenizer->merges, n_capacity) != GPT2_OK) {
    gpt2_hashmap_free(&tokenizer->tokens);
    gpt2_map_free(&map);
    return GPT2_ERR_INVALID;
  }
  gpt2_init_byte_maps(tokenizer);
  json.ptr = map.ptr, json.pos = 0, json.len = map.size;
  if (gpt2_json_skip(&json) != '{') {
    gpt2_hashmap_free(&tokenizer->tokens);
    gpt2_hashmap_free(&tokenizer->merges);
    gpt2_map_free(&map);
    return GPT2_ERR_INVALID;
  }
  gpt2_json_next(&json);
  while (1) {
    c = gpt2_json_skip(&json);
    if (c == EOF) break;
    if (state == gpt2_parse_object) {
      if (c == '}') break;
      if (gpt2_json_get_raw_string(&json, &key)) break;
      if (gpt2_json_skip(&json) != ':') break;
      gpt2_json_next(&json);
      if (gpt2_string_equal(key, gpt2_string_t("added_tokens"))) {
        if (gpt2_json_skip(&json) != '[') {
          gpt2_json_skip_value(&json);
        } else {
          gpt2_json_next(&json);
          state = gpt2_parse_add_token;
        }
      } else if (gpt2_string_equal(key, gpt2_string_t("vocab"))) {
        if (gpt2_json_skip(&json) != '{') {
          ok = -1;
          break;
        } else {
          gpt2_json_next(&json);
          state = gpt2_parse_vocab;
        }
      } else if (gpt2_string_equal(key, gpt2_string_t("merges"))) {
        if (gpt2_json_skip(&json) != '[') {
          ok = -1;
          break;
        } else {
          gpt2_json_next(&json);
          state = gpt2_parse_merge;
        }
      } else if (gpt2_string_equal(key, gpt2_string_t("model"))) {
        if (gpt2_json_skip(&json) == '{') {
          gpt2_json_next(&json);
        } else {
          gpt2_json_skip_value(&json);
        }
      } else {
        gpt2_json_skip_value(&json);
      }
    } else if (state == gpt2_parse_add_token) {
      c = gpt2_json_skip(&json);
      if (c == EOF) break;
      if (c == '{') {
        gpt2_json_next(&json);
      }
      if (gpt2_json_get_raw_string(&json, &key)) break;
      if (gpt2_json_skip(&json) != ':') break;
      gpt2_json_next(&json);
      if (gpt2_string_equal(key, gpt2_string_t("id"))) {
        if (gpt2_json_get_uint64(&json, &current_id)) break;
      } else if (gpt2_string_equal(key, gpt2_string_t("content"))) {
        if (gpt2_json_get_string(&json, &sval)) break;
      } else {
        gpt2_json_skip_value(&json);
      }
      c = gpt2_json_skip(&json);
      if (c == '}') {
        gpt2_json_next(&json);
        c = gpt2_json_skip(&json);
        if (current_id != (uint64_t)-1 && sval.len > 0) {
          if (gpt2_hashmap_put(
            &tokenizer->tokens, &sval, (int)current_id
          ) == GPT2_OK) {
            if (config) {
              if (
                gpt2_string_equal(sval, gpt2_string_t("<s>")) ||
                gpt2_string_equal(sval, gpt2_string_t("<|startoftext|>"))
              ) {
                if (config->bos_token_id < 0) {
                  config->bos_token_id = (int)current_id;
                }
              } else if (
                gpt2_string_equal(sval, gpt2_string_t("</s>")) ||
                gpt2_string_equal(sval, gpt2_string_t("<|endoftext|>"))
              ) {
                if (config->eos_token_id < 0) {
                  config->eos_token_id = (int)current_id;
                }
              } else if (gpt2_string_equal(sval, gpt2_string_t("<pad>"))) {
                if (config->pad_token_id < 0) {
                  config->pad_token_id = (int)current_id;
                }
              } else if (gpt2_string_equal(sval, gpt2_string_t("<unk>"))) {
                if (config->unk_token_id < 0) {
                  config->unk_token_id = (int)current_id;
                }
              }
            }
            sval.buf = NULL, sval.len = 0;
          } else {
            printf("Add '%.*s' %d failed\n", (int)sval.len, sval.buf, (int)current_id);
          }
        }
        current_id = (uint64_t)-1;
        gpt2_string_free(&sval);
      }
      if (c == ']') {
        gpt2_json_next(&json);
        state = gpt2_parse_object;
      }
    } else if (state == gpt2_parse_vocab) {
      struct gpt2_string_t val = {0};
      uint64_t id = 0;
      if (gpt2_json_get_string(&json, &val)) break;
      if (gpt2_json_skip(&json) != ':') {
        gpt2_string_free(&val);
        break;
      }
      gpt2_json_next(&json);
      if (gpt2_json_get_uint64(&json, &id)) {
        gpt2_string_free(&val);
        break;
      }
      if (gpt2_hashmap_put(&tokenizer->tokens, &val, (int)id) == GPT2_OK) {
        val.buf = NULL, val.len = 0;
      } else {
        printf("Add '%.*s' %d failed\n", (int)val.len, val.buf, (int)id);
      }
      gpt2_string_free(&val);
      c = gpt2_json_skip(&json);
      if (c == '}') {
        gpt2_json_next(&json);
        state = gpt2_parse_object;
      }
    } else if (state == gpt2_parse_merge) {
      c = gpt2_json_skip(&json);
      if (c == EOF) break;
      if (c == '[') gpt2_json_next(&json);
      struct gpt2_string_t left = {0}, right = {0};
      if (gpt2_json_get_string(&json, &left)) break;
      if (gpt2_json_skip(&json) != ',') {
        gpt2_string_free(&left);
        break;
      }
      gpt2_json_next(&json);
      if (gpt2_json_get_string(&json, &right)) {
        gpt2_string_free(&left);
        break;
      }
      rank++;
      if (gpt2_string_append(&left, right.buf, right.len) == 0) {
        if (gpt2_hashmap_put(&tokenizer->merges, &left, (int)rank) == GPT2_OK) {
          left.buf = NULL, left.len = 0;
        } else {
          printf("Add '%.*s' %d failed\n", (int)left.len, left.buf, (int)rank);
        }
      }

      gpt2_string_free(&left);
      gpt2_string_free(&right);

      if (gpt2_json_skip(&json) != ']') break;
      gpt2_json_next(&json);
      c = gpt2_json_skip(&json);
      if (c == ',') gpt2_json_next(&json);
      if (c == ']') {
        gpt2_json_next(&json);
        state = gpt2_parse_object;
      }
    }
    if (gpt2_json_skip(&json) == ',') {
      gpt2_json_next(&json);
    }
  }
  gpt2_map_free(&map);
  if (ok != GPT2_OK) {
    gpt2_hashmap_free(&tokenizer->tokens);
    gpt2_hashmap_free(&tokenizer->merges);
  } else {
    tokenizer->id2token = (unsigned int *)calloc(
      tokenizer->tokens.size, sizeof(unsigned int)
    );
    if (!tokenizer->id2token) {
      gpt2_hashmap_free(&tokenizer->tokens);
      gpt2_hashmap_free(&tokenizer->merges);
    } else {
      size_t i;
      for (i = 0; i < tokenizer->tokens.capacity; i++) {
        if (tokenizer->tokens.buckets[i].state == gpt2_cell_occupied) {
          tokenizer->id2token[tokenizer->tokens.buckets[i].value] = i;
        }
      }
    }
  }
  return ok;
}

static int gpt2_load_tokenize_config(
  const char *path, const struct gpt2_tokenize_t *tokenizer,
  struct gpt2_config_t *config
) {
  int c, ok = GPT2_OK;
  struct gpt2_map_t map = {0};
  struct gpt2_json_t json = {0};
  if (!tokenizer || !config) return -1;
  if (gpt2_map_init(path, &map)) {
    return -1;
  }
  json.ptr = map.ptr, json.pos = 0, json.len = map.size;
  c = gpt2_json_skip(&json);
  if (c != '{') {
    gpt2_map_free(&map);
    return GPT2_ERR_INVALID;
  }
  gpt2_json_next(&json);
  while (1) {
    c = gpt2_json_skip(&json);
    if (c == '}' || c == EOF) {
      gpt2_json_next(&json);
      break;
    }
    struct gpt2_string_t key = {NULL, 0};
    if (gpt2_json_get_raw_string(&json, &key)) {
      ok = GPT2_ERR_INVALID;
      break;
    }
    c = gpt2_json_skip(&json);
    if (c != ':') {
      ok = GPT2_ERR_INVALID;
      break;
    }
    gpt2_json_next(&json);
    if (gpt2_string_equal(key, gpt2_string_t("bos_token"))) {
      if (gpt2_json_skip(&json) == 'n') {
        if (gpt2_json_skip_value(&json)) {
          ok = GPT2_ERR_INVALID;
          break;
        }
        config->bos_token_id = -1;
      } else {
        if (gpt2_json_get_string(&json, &key) == 0) {
          int *token_id = gpt2_hashmap_get(&tokenizer->tokens, &key);
          if (token_id) {
            config->bos_token_id = *token_id;
          }
          gpt2_string_free(&key);
        }
      }
    } else if (gpt2_string_equal(key, gpt2_string_t("eos_token"))) {
      if (gpt2_json_skip(&json) == 'n') {
        if (gpt2_json_skip_value(&json)) {
          ok = GPT2_ERR_INVALID;
          break;
        }
        config->eos_token_id = -1;
      } else {
        if (gpt2_json_get_string(&json, &key) == 0) {
          int *token_id = gpt2_hashmap_get(&tokenizer->tokens, &key);
          if (token_id) {
            config->eos_token_id = *token_id;
          } else {
            config->eos_token_id = -1;
          }
          gpt2_string_free(&key);
        }
      }
    } else if (gpt2_string_equal(key, gpt2_string_t("pad_token"))) {
      if (gpt2_json_skip(&json) == 'n') {
        if (gpt2_json_skip_value(&json)) {
          ok = GPT2_ERR_INVALID;
          break;
        }
        config->pad_token_id = -1;
      } else {
        if (gpt2_json_get_string(&json, &key) == 0) {
          int *token_id = gpt2_hashmap_get(&tokenizer->tokens, &key);
          if (token_id) {
            config->pad_token_id = *token_id;
          } else {
            config->pad_token_id = -1;
          }
          gpt2_string_free(&key);
        }
      }
    } else if (gpt2_string_equal(key, gpt2_string_t("unk_token"))) {
      if (gpt2_json_skip(&json) == 'n') {
        if (gpt2_json_skip_value(&json)) {
          ok = GPT2_ERR_INVALID;
          break;
        }
        config->unk_token_id = -1;
      } else {
        if (gpt2_json_get_string(&json, &key) == 0) {
          int *token_id = gpt2_hashmap_get(&tokenizer->tokens, &key);
          if (token_id) {
            config->unk_token_id = *token_id;
          } else {
            config->unk_token_id = -1;
          }
          gpt2_string_free(&key);
        }
      }
    } else {
      if (gpt2_json_skip_value(&json)) {
        ok = GPT2_ERR_INVALID;
        break;
      }
    }
    c = gpt2_json_skip(&json);
    if (c == ',') {
      gpt2_json_next(&json);
    } else if (c != '}') {
      ok = GPT2_OK;
      break;
    }
  }
  gpt2_map_free(&map);
  return ok;
}

/* Encode / Decoded */
static int utf8_char_length(unsigned char c) {
  if ((c & 0x80) == 0x00) return 1;
  if ((c & 0xE0) == 0xC0) return 2;
  if ((c & 0xF0) == 0xE0) return 3;
  if ((c & 0xF8) == 0xF0) return 4;
  return 1;
}

static int is_space(uint32_t cp) {
  return cp == ' ' || (cp >= 0x09 && cp <= 0x0D) || cp == 0x85 || cp == 0xA0;
}

static int is_number(uint32_t cp) {
  return cp >= '0' && cp <= '9';
}

static int is_letter(uint32_t cp) {
  if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) return 1;
  if (cp >= 0x80 && !is_space(cp)) return 1;
  return 0;
}

static int match_contraction(const char *input, size_t len, size_t i) {
  if (i >= len) return 0;
  char c = input[i];
  size_t quote_len = 0;

  if (c == '\'' || c == '`') {
    quote_len = 1;
  } else if (
    (unsigned char)input[i] == 0xE2 && i + 2 < len &&
    (unsigned char)input[i+1] == 0x80 &&
    (
      (unsigned char)input[i+2] == 0x99 ||
      (unsigned char)input[i+2] == 0x98
    )
  ) {
    quote_len = 3;
  }

  if (quote_len == 0) return 0;

  size_t next_i = i + quote_len;
  if (next_i >= len) return 0;

  int next_char = tolower((unsigned char)input[next_i]);
  size_t suffix_len = 0;

  if (next_char == 's' || next_char == 't' || next_char == 'm' || next_char == 'd') {
    if (next_i + 1 >= len || !is_letter((unsigned char)input[next_i + 1])) {
      suffix_len = quote_len + 1;
    }
  } else if (next_char == 'r' && next_i + 1 < len && tolower((unsigned char)input[next_i + 1]) == 'e') {
    if (next_i + 2 >= len || !is_letter((unsigned char)input[next_i + 2])) {
      suffix_len = quote_len + 2;
    }
  } else if (next_char == 'v' && next_i + 1 < len && tolower((unsigned char)input[next_i + 1]) == 'e') {
    if (next_i + 2 >= len || !is_letter((unsigned char)input[next_i + 2])) {
      suffix_len = quote_len + 2;
    }
  } else if (next_char == 'l' && next_i + 1 < len && tolower((unsigned char)input[next_i + 1]) == 'l') {
    if (next_i + 2 >= len || !is_letter((unsigned char)input[next_i + 2])) {
      suffix_len = quote_len + 2;
    }
  }
  return (int)suffix_len;
}

static int gpt2_pretokenize(const char *input, struct gpt2_string_t **tokens) {
  if (!input || !tokens) return 0;

  size_t len = strlen(input);
  if (len == 0) {
    *tokens = NULL;
    return 0;
  }

  size_t capacity = 16;
  size_t count = 0;

  struct gpt2_string_t *list = malloc(capacity * sizeof(struct gpt2_string_t));
  if (!list) return -1;

  size_t i = 0;

  while (i < len) {
    size_t start = i;
    unsigned char first_char = (unsigned char)input[i];
    int char_len = utf8_char_length(first_char);

    uint32_t cp = first_char;
    if (char_len == 2) {
      cp = ((first_char & 0x1F) << 6) | (input[i+1] & 0x3F);
    } else if (char_len == 3) {
      cp = ((first_char & 0x0F) << 12) | ((input[i+1] & 0x3F) << 6) | (input[i+2] & 0x3F);
    } else if (char_len == 4) {
      cp = ((first_char & 0x07) << 18) | ((input[i+1] & 0x3F) << 12) | ((input[i+2] & 0x3F) << 6) | (input[i+3] & 0x3F);
    }

    if (is_space(cp)) {
      while (i < len) {
        int clen = utf8_char_length((unsigned char)input[i]);
        uint32_t next_cp = (unsigned char)input[i];
        if (clen == 2) next_cp = (((unsigned char)input[i] & 0x1F) << 6) | ((unsigned char)input[i+1] & 0x3F);
        if (!is_space(next_cp)) break;
        i += clen;
      }
    }
    else if (is_number(cp)) {
      int has_dot = 0;
      while (i < len) {
        int clen = utf8_char_length((unsigned char)input[i]);
        if (clen != 1) break;
        char c = input[i];
        if (is_number(c)) {
          i += clen;
        } else if (c == '.' && !has_dot && i + 1 < len && is_number((unsigned char)input[i + 1])) {
          has_dot = 1;
          i += clen;
        } else {
          break;
        }
      }
    }
    else if (is_letter(cp)) {
      while (i < len) {
        int cont_len = match_contraction(input, len, i);
        if (cont_len > 0) {
          i += cont_len;
          break;
        }
        int clen = utf8_char_length((unsigned char)input[i]);
        uint32_t next_cp = (unsigned char)input[i];
        if (clen == 2) next_cp = (((unsigned char)input[i] & 0x1F) << 6) | ((unsigned char)input[i+1] & 0x3F);
        else if (clen >= 3) next_cp = 0x80;
        if (!is_letter(next_cp)) break;
        i += clen;
      }
    }
    else {
      i += char_len;
    }

    size_t token_len = i - start;
    if (token_len > 0) {
      if (count >= capacity) {
        capacity *= 2;
        struct gpt2_string_t *temp = realloc(list, capacity * sizeof(struct gpt2_string_t));
        if (!temp) {
          free(list);
          return -1;
        }
        list = temp;
      }

      list[count].buf = (char *)&input[start];
      list[count].len = token_len;
      count++;
    }
  }

  *tokens = list;
  return (int)count;
}

static int gpt2_tokenize_encode(struct gpt2_tokenize_t *tokenizer, const char *prompt, int **tokens) {
  if (!tokenizer || !prompt || !tokens) return GPT2_ERR_INVALID;

  struct gpt2_string_t *pre_tokens = NULL;
  int pre_count = gpt2_pretokenize(prompt, &pre_tokens);
  if (pre_count < 0) return GPT2_ERR_NOMEM;
  if (pre_count == 0) {
    *tokens = NULL;
    return 0;
  }

  struct gpt2_bpe_node_t *head = NULL, *tail = NULL;
  size_t total_tokens = 0;

  for (int p = 0; p < pre_count; p++) {
    const char *p_buf = pre_tokens[p].buf;
    size_t p_len = pre_tokens[p].len;
    size_t i = 0;

    while (i < p_len) {
      unsigned char byte_val = (unsigned char)p_buf[i];
      char *mapped_char = tokenizer->byte_encoder[byte_val];
      size_t m_len = strlen(mapped_char);

      struct gpt2_bpe_node_t *node = (struct gpt2_bpe_node_t *)malloc(sizeof(struct gpt2_bpe_node_t));
      if (!node) {
        free(pre_tokens);
        while (head) {
          struct gpt2_bpe_node_t *temp = head;
          head = head->next;
          free(temp->token.buf);
          free(temp);
        }
        return GPT2_ERR_NOMEM;
      }

      node->token.buf = (char *)malloc(m_len);
      memcpy(node->token.buf, mapped_char, m_len);
      node->token.len = m_len;
      node->next = NULL;

      if (!head) {
        head = node;
      } else {
        tail->next = node;
      }
      tail = node;
      total_tokens++;

      i++;
    }
  }
  free(pre_tokens);

  while (1) {
    struct gpt2_bpe_node_t *curr = head;
    struct gpt2_bpe_node_t *best_prev = NULL;
    int min_rank = INT_MAX;

    while (curr && curr->next) {
      int *rank_ptr = gpt2_hashmap_get_merge(&tokenizer->merges, &curr->token, &curr->next->token);
      if (rank_ptr && *rank_ptr < min_rank) {
        min_rank = *rank_ptr;
        best_prev = curr;
      }
      curr = curr->next;
    }

    if (min_rank == INT_MAX || !best_prev) {
      break;
    }

    struct gpt2_bpe_node_t *first = best_prev;
    struct gpt2_bpe_node_t *second = best_prev->next;

    size_t new_len = first->token.len + second->token.len;
    char *new_buf = (char *)malloc(new_len);
    if (!new_buf) break;

    memcpy(new_buf, first->token.buf, first->token.len);
    memcpy(new_buf + first->token.len, second->token.buf, second->token.len);

    free(first->token.buf);
    first->token.buf = new_buf;
    first->token.len = new_len;

    first->next = second->next;
    free(second->token.buf);
    free(second);

    total_tokens--;
  }

  int *res_tokens = (int *)malloc(total_tokens * sizeof(int));
  if (!res_tokens) {
    while (head) {
      struct gpt2_bpe_node_t *temp = head;
      head = head->next;
      free(temp->token.buf);
      free(temp);
    }
    return GPT2_ERR_NOMEM;
  }

  struct gpt2_bpe_node_t *curr = head;
  size_t idx = 0;
  while (curr) {
    int *token_id = gpt2_hashmap_get(&tokenizer->tokens, &curr->token);
    if (token_id) {
      res_tokens[idx++] = *token_id;
    } else {
      res_tokens[idx++] = 0;
    }

    struct gpt2_bpe_node_t *temp = curr;
    curr = curr->next;
    free(temp->token.buf);
    free(temp);
  }

  *tokens = res_tokens;
  return (int)total_tokens;
}

static int gpt2_tokenize_decode(struct gpt2_tokenize_t *tokenizer, const int *tokens, int ntok, char **out) {
  if (!tokenizer || !tokens || ntok <= 0 || !out) return GPT2_ERR_INVALID;

  struct gpt2_string_t raw_str = {NULL, 0};

  for (int i = 0; i < ntok; i++) {
    struct gpt2_string_t tok_val = {NULL, 0};
    if (gpt2_get_token(tokenizer, tokens[i], &tok_val) == 0) {
      if (gpt2_string_append(&raw_str, tok_val.buf, tok_val.len) != 0) {
        gpt2_string_free(&raw_str);
        return GPT2_ERR_NOMEM;
      }
    }
  }

  if (!raw_str.buf || raw_str.len == 0) {
    *out = (char *)malloc(1);
    if (*out) (*out)[0] = '\0';
    gpt2_string_free(&raw_str);
    return 0;
  }

  size_t decoded_capacity = raw_str.len + 1;
  char *decoded_buf = (char *)malloc(decoded_capacity);
  if (!decoded_buf) {
    gpt2_string_free(&raw_str);
    return GPT2_ERR_NOMEM;
  }

  size_t r_idx = 0;
  size_t w_idx = 0;

  while (r_idx < raw_str.len) {
    unsigned char c = (unsigned char)raw_str.buf[r_idx];
    int char_len = 1;
    unsigned int cp = c;

    if ((c & 0xE0) == 0xC0 && r_idx + 1 < raw_str.len) {
      char_len = 2;
      cp = ((c & 0x1F) << 6) | ((unsigned char)raw_str.buf[r_idx + 1] & 0x3F);
    } else if ((c & 0xF0) == 0xE0 && r_idx + 2 < raw_str.len) {
      char_len = 3;
      cp = ((c & 0x0F) << 12) | (((unsigned char)raw_str.buf[r_idx + 1] & 0x3F) << 6) | ((unsigned char)raw_str.buf[r_idx + 2] & 0x3F);
    }

    if (cp < 512 && tokenizer->byte_decoder[cp] != 0) {
      decoded_buf[w_idx++] = (char)tokenizer->byte_decoder[cp];
    } else {
      for (int k = 0; k < char_len && (r_idx + k) < raw_str.len; k++) {
        decoded_buf[w_idx++] = raw_str.buf[r_idx + k];
      }
    }
    r_idx += char_len;
  }

  decoded_buf[w_idx] = '\0';
  gpt2_string_free(&raw_str);

  *out = decoded_buf;
  return (int)w_idx;
}
/* Encode / Decoded */

static int gpt2_load_safetensor(const char *path, struct gpt2_model_t *model) {
  if (!path || !model) return -1;

  if (gpt2_map_init(path, &model->map) != 0) {
    return -1;
  }

  if (model->map.size < 8) {
    gpt2_map_free(&model->map);
    return -1;
  }

  uint64_t header_len = 0;
  memcpy(&header_len, model->map.ptr, sizeof(uint64_t));

  if (8 + header_len > model->map.size) {
    gpt2_map_free(&model->map);
    return -1;
  }

  unsigned char *weights_base_ptr = model->map.ptr + 8 + header_len;

  struct gpt2_json_t json;
  json.ptr = model->map.ptr + 8;
  json.pos = 0;
  json.len = (size_t)header_len;

  int c = gpt2_json_skip(&json);
  if (c != '{') {
    gpt2_map_free(&model->map);
    return -1;
  }
  gpt2_json_next(&json);

  int max_layer_idx = -1;
  int estimated_layers = model->config.n_layer;
  model->layers = (struct gpt2_weight_t *)calloc(estimated_layers, sizeof(struct gpt2_weight_t));
  if (!model->layers) {
    gpt2_map_free(&model->map);
    return -1;
  }

  while (1) {
    c = gpt2_json_skip(&json);
    if (c == '}' || c == EOF) break;
    if (c == ',') {
      gpt2_json_next(&json);
      continue;
    }
    struct gpt2_string_t raw_key_str;
    if (gpt2_json_get_raw_string(&json, &raw_key_str) != 0) {
      break;
    }

    struct gpt2_string_t key_str = raw_key_str;
    const char *transformer_prefix = "transformer.";
    size_t prefix_len = strlen(transformer_prefix);
    if (key_str.len > prefix_len && strncmp(key_str.buf, transformer_prefix, prefix_len) == 0) {
      key_str.buf += prefix_len;
      key_str.len -= prefix_len;
    }

    c = gpt2_json_skip(&json);
    if (c == ':') gpt2_json_next(&json);

    c = gpt2_json_skip(&json);
    if (c == '{') {
      gpt2_json_next(&json);

      enum gpt2_dtype_t dtype = gpt2_dtype_unknown;
      int shape[4] = {0};
      int ndim = 0;
      uint64_t start_offset = 0, end_offset = 0;

      while (1) {
        int inner_c = gpt2_json_skip(&json);
        if (inner_c == '}' || inner_c == EOF) {
          if (inner_c == '}') gpt2_json_next(&json);
          break;
        }
        if (inner_c == ',') {
          gpt2_json_next(&json);
          continue;
        }

        struct gpt2_string_t prop_key;
        if (gpt2_json_get_raw_string(&json, &prop_key) != 0) break;

        inner_c = gpt2_json_skip(&json);
        if (inner_c == ':') gpt2_json_next(&json);

        if (gpt2_string_equal(prop_key, gpt2_string_s("dtype"))) {
          struct gpt2_string_t dtype_str;
          gpt2_json_get_raw_string(&json, &dtype_str);
          if (gpt2_string_equal(dtype_str, gpt2_string_s("F32"))) dtype = gpt2_dtype_f32;
          else if (gpt2_string_equal(dtype_str, gpt2_string_s("F16"))) dtype = gpt2_dtype_f16;
          else if (gpt2_string_equal(dtype_str, gpt2_string_s("BF16"))) dtype = gpt2_dtype_bf16;
          else if (gpt2_string_equal(dtype_str, gpt2_string_s("I32"))) dtype = gpt2_dtype_i32;
          else if (gpt2_string_equal(dtype_str, gpt2_string_s("I64"))) dtype = gpt2_dtype_i64;
        } else if (gpt2_string_equal(prop_key, gpt2_string_s("shape"))) {
          inner_c = gpt2_json_skip(&json);
          if (inner_c == '[') {
            gpt2_json_next(&json);
            while (1) {
              inner_c = gpt2_json_skip(&json);
              if (inner_c == ']' || inner_c == EOF) {
                if (inner_c == ']') gpt2_json_next(&json);
                  break;
                }
                if (inner_c == ',') {
                  gpt2_json_next(&json);
                  continue;
                }
                uint64_t dim_val = 0;
                gpt2_json_get_uint64(&json, &dim_val);
                if (ndim < 4) {
                  shape[ndim++] = (int)dim_val;
                }
              }
            }
          } else if (gpt2_string_equal(prop_key, gpt2_string_s("data_offsets"))) {
            inner_c = gpt2_json_skip(&json);
            if (inner_c == '[') {
              gpt2_json_next(&json);
              gpt2_json_get_uint64(&json, &start_offset);
              inner_c = gpt2_json_skip(&json);
              if (inner_c == ',') gpt2_json_next(&json);
              gpt2_json_get_uint64(&json, &end_offset);
              inner_c = gpt2_json_skip(&json);
              if (inner_c == ']') gpt2_json_next(&json);
            }
          } else {
            gpt2_json_skip_value(&json);
          }
        }

        struct gpt2_tensor_t loaded_tensor;
        loaded_tensor.dtype = dtype;
        loaded_tensor.ndim = ndim;
        for (int i = 0; i < 4; i++) loaded_tensor.shape[i] = shape[i];
        loaded_tensor.start_offset = start_offset;
        loaded_tensor.end_offset = end_offset;
        loaded_tensor.data = (void *)(weights_base_ptr + start_offset);

        if (gpt2_string_equal(key_str, gpt2_string_s("wte.weight"))) {
          model->wte = loaded_tensor;
        } else if (gpt2_string_equal(key_str, gpt2_string_s("lm_head.weight"))) {
          model->lm_head = loaded_tensor;
        } else if (gpt2_string_equal(key_str, gpt2_string_s("wpe.weight"))) {
          model->wpe = loaded_tensor;
        } else if (gpt2_string_equal(key_str, gpt2_string_s("ln_f.weight"))) {
          model->ln_f_w = loaded_tensor;
        } else if (gpt2_string_equal(key_str, gpt2_string_s("ln_f.bias"))) {
          model->ln_f_b = loaded_tensor;
        } else if (key_str.len > 2 && key_str.buf[0] == 'h' && key_str.buf[1] == '.') {
          int layer_idx = 0;
          size_t idx = 2;
          while (idx < key_str.len && isdigit((unsigned char)key_str.buf[idx])) {
            layer_idx = layer_idx * 10 + (key_str.buf[idx] - '0');
            idx++;
          }
          if (layer_idx > max_layer_idx) {
            max_layer_idx = layer_idx;
          }
          if (layer_idx >= estimated_layers) {
            estimated_layers = layer_idx + 10;
            struct gpt2_weight_t *new_layers = realloc(
              model->layers, estimated_layers * sizeof(struct gpt2_weight_t)
            );
            if (!new_layers) {
              free(model->layers), model->layers = NULL;
              gpt2_map_free(&model->map);
              return -1;
            }
            model->layers = new_layers;
          }
          if (idx < key_str.len && key_str.buf[idx] == '.') {
            const char *sub_name = key_str.buf + idx + 1;
            size_t sub_len = key_str.len - (idx + 1);
            struct gpt2_string_t sub_str = gpt2_string_n(sub_name, sub_len);
            struct gpt2_weight_t *layer = &model->layers[layer_idx];

            if (gpt2_string_equal(sub_str, gpt2_string_s("ln_1.weight"))) {
              layer->ln_1_w = loaded_tensor;
            } else if (gpt2_string_equal(sub_str, gpt2_string_s("ln_1.bias"))) {
              layer->ln_1_b = loaded_tensor;
            } else if (gpt2_string_equal(sub_str, gpt2_string_s("attn.c_attn.weight"))) {
              layer->qkv_w = loaded_tensor;
            } else if (gpt2_string_equal(sub_str, gpt2_string_s("attn.c_attn.bias"))) {
              layer->qkv_b = loaded_tensor;
            } else if (gpt2_string_equal(sub_str, gpt2_string_s("attn.c_proj.weight"))) {
              layer->attn_proj_w = loaded_tensor;
            } else if (gpt2_string_equal(sub_str, gpt2_string_s("attn.c_proj.bias"))) {
              layer->attn_proj_b = loaded_tensor;
            } else if (gpt2_string_equal(sub_str, gpt2_string_s("ln_2.weight"))) {
              layer->ln_2_w = loaded_tensor;
            } else if (gpt2_string_equal(sub_str, gpt2_string_s("ln_2.bias"))) {
              layer->ln_2_b = loaded_tensor;
            } else if (gpt2_string_equal(sub_str, gpt2_string_s("mlp.c_fc.weight"))) {
              layer->fc_w = loaded_tensor;
            } else if (gpt2_string_equal(sub_str, gpt2_string_s("mlp.c_fc.bias"))) {
              layer->fc_b = loaded_tensor;
            } else if (gpt2_string_equal(sub_str, gpt2_string_s("mlp.c_proj.weight"))) {
              layer->proj_w = loaded_tensor;
            } else if (gpt2_string_equal(sub_str, gpt2_string_s("mlp.c_proj.bias"))) {
              layer->proj_b = loaded_tensor;
            }
          }
        }
      } else {
        gpt2_json_skip_value(&json);
      }
  }
  if (model->config.tie_word_embeddings) {
    if (!model->lm_head.data && model->wte.data) {
      model->lm_head = model->wte;
    }
  }

  model->config.n_layer = max_layer_idx + 1;

  if (
    !model->wte.data || !model->wpe.data ||
    !model->ln_f_w.data || !model->ln_f_b.data ||
    (!model->lm_head.data && !model->config.tie_word_embeddings)
  ) {
    free(model->layers), model->layers = NULL;
    gpt2_map_free(&model->map);
    return -1;
  }

  for (int i = 0; i < model->config.n_layer; i++) {
    struct gpt2_weight_t *layer = &model->layers[i];
    if (
      !layer->ln_1_w.data || !layer->ln_1_b.data ||
      !layer->qkv_w.data || !layer->qkv_b.data ||
      !layer->attn_proj_w.data || !layer->attn_proj_b.data ||
      !layer->ln_2_w.data || !layer->ln_2_b.data ||
      !layer->fc_w.data || !layer->fc_b.data ||
      !layer->proj_w.data || !layer->proj_b.data
    ) {
      free(model->layers), model->layers = NULL;
      gpt2_map_free(&model->map);
      return -1;
    }
  }

  return 0;
}

int gpt2_load_model(const char *path, struct gpt2_model_t *model) {
  char fpath[PATH_MAX] = {0};
  if (!path || !model) {
    return -1;
  }
  if (snprintf(fpath, sizeof(fpath), "%s/config.json", path) <= 0) {
    return -1;
  }
  if (gpt2_load_config(fpath, &model->config) != GPT2_OK) {
    return -1;
  }
  if (snprintf(fpath, sizeof(fpath), "%s/tokenizer.json", path) <= 0) {
    return -1;
  }
  if (gpt2_load_tokenize(fpath, &model->tokenize, &model->config) != GPT2_OK) {
    return -1;
  }
  if (snprintf(fpath, sizeof(fpath), "%s/tokenizer_config.json", path) > 0) {
    if (gpt2_load_tokenize_config(fpath, &model->tokenize, &model->config) != GPT2_OK) {
      printf("Load config 'fpath' failed\n");
    }
  }
  if (snprintf(fpath, sizeof(fpath), "%s/model.safetensors", path) <= 0) {
    gpt2_hashmap_free(&model->tokenize.tokens);
    gpt2_hashmap_free(&model->tokenize.merges);
    if (model->tokenize.id2token) {
      free(model->tokenize.id2token);
      model->tokenize.id2token = NULL;
    }
    return -1;
  }
  if (gpt2_load_safetensor(fpath, model) != GPT2_OK) {
    gpt2_hashmap_free(&model->tokenize.tokens);
    gpt2_hashmap_free(&model->tokenize.merges);
    if (model->tokenize.id2token) {
      free(model->tokenize.id2token);
      model->tokenize.id2token = NULL;
    }
    return -1;
  }
  return GPT2_OK;
}

void gpt2_model_free(struct gpt2_model_t *model) {
  if (!model) return;
  gpt2_hashmap_free(&model->tokenize.tokens);
  gpt2_hashmap_free(&model->tokenize.merges);
  if (model->tokenize.id2token) {
    free(model->tokenize.id2token);
    model->tokenize.id2token = NULL;
  }
  if (model->layers) {
    free(model->layers);
    model->layers = NULL;
  }
  gpt2_map_free(&model->map);
  model->wte.data = NULL;
  model->wpe.data = NULL;
  model->ln_f_w.data = NULL;
  model->ln_f_b.data = NULL;
  memset(model, 0, sizeof(struct gpt2_model_t));
}

int gpt2_token_decode_single(struct gpt2_tokenize_t *tokenizer, int token, struct gpt2_string_t *out) {
  if (!tokenizer || !out) return GPT2_ERR_INVALID;

  struct gpt2_string_t tok_val = {NULL, 0};
  if (gpt2_get_token(tokenizer, token, &tok_val) != 0) {
    return GPT2_ERR_INVALID;
  }

  if (!tok_val.buf || tok_val.len == 0) {
    out->buf = (char *)malloc(1);
    if (!out->buf) return GPT2_ERR_NOMEM;
    out->buf[0] = '\0';
    out->len = 0;
    return 0;
  }

  size_t decoded_capacity = tok_val.len + 1;
  char *decoded_buf = (char *)malloc(decoded_capacity);
  if (!decoded_buf) {
    return GPT2_ERR_NOMEM;
  }

  size_t r_idx = 0;
  size_t w_idx = 0;

  while (r_idx < tok_val.len) {
    unsigned char c = (unsigned char)tok_val.buf[r_idx];
    int char_len = 1;
    unsigned int cp = c;

    if ((c & 0xE0) == 0xC0 && r_idx + 1 < tok_val.len) {
      char_len = 2;
      cp = ((c & 0x1F) << 6) | ((unsigned char)tok_val.buf[r_idx + 1] & 0x3F);
    } else if ((c & 0xF0) == 0xE0 && r_idx + 2 < tok_val.len) {
      char_len = 3;
      cp = ((c & 0x0F) << 12) | (((unsigned char)tok_val.buf[r_idx + 1] & 0x3F) << 6) | ((unsigned char)tok_val.buf[r_idx + 2] & 0x3F);
    }

    if (cp < 512 && tokenizer->byte_decoder[cp] != 0) {
      decoded_buf[w_idx++] = (char)tokenizer->byte_decoder[cp];
    } else {
      for (int k = 0; k < char_len && (r_idx + k) < tok_val.len; k++) {
        decoded_buf[w_idx++] = tok_val.buf[r_idx + k];
      }
    }
    r_idx += char_len;
  }

  decoded_buf[w_idx] = '\0';

  out->buf = decoded_buf;
  out->len = w_idx;

  return (int)w_idx;
}

int gpt2_model_decode_single(struct gpt2_model_t *model, int token, char **out) {
  if (model && out) {
    struct gpt2_string_t tok = {NULL, 0};
    if (gpt2_token_decode_single(&model->tokenize, token, &tok) > 0) {
      *out = tok.buf;
      return (int)tok.len;
    }
  }
  return -1;
}

int gpt2_model_encode(struct gpt2_model_t *model, const char *prompt, int **tokens) {
  if (!model) return -1;
  return gpt2_tokenize_encode(&model->tokenize, prompt, tokens);
}

int gpt2_model_decode(struct gpt2_model_t *model, const int *tokens, int ntok, char **out) {
  if (!model) return -1;
  return gpt2_tokenize_decode(&model->tokenize, tokens, ntok, out);
}

static int gpt2_float_resize(struct gpt2_float_t *buf, int size) {
  if (buf && size > 0) {
    size_t required_capacity = buf->pos + (size_t)size;
    if (required_capacity > buf->capacity) {
      float *new_data = (float *)realloc(buf->data, required_capacity * sizeof(float));
      if (!new_data) return -1;
      buf->data = new_data, buf->capacity = required_capacity;
    }
    return 0;
  }
  if (size == 0) return 0;
  return -1;
}

static void gpt2_float_free(struct gpt2_float_t *buf) {
  if (buf) {
    if (buf->data) free(buf->data);
    buf->data = NULL, buf->pos = 0, buf->capacity = 0;
  }
}

int gpt2_state_init(struct gpt2_model_t *model) {
  struct gpt2_state_t *kv = NULL;
  if (model) {
    size_t n_embd = model->config.n_embd;
    size_t vocab_size = model->config.vocab_size;
    size_t n_positions = model->config.n_positions;
    size_t n_layer = model->config.n_layer;
    size_t size_kv_total = n_layer * n_positions * n_embd;

    kv = &model->state;

    kv->x = kv->logits = kv->final = NULL;
    kv->keys = kv->values = NULL;
    kv->qkv = kv->att_scores = kv->attn_out = NULL;
    kv->resid = kv->ln1 = kv->ln2 = kv->mlp_hidden = kv->mlp_out = NULL;
    kv->seq_len = 0;

    kv->x = (float *)malloc(n_embd * sizeof(float));
    if (!kv->x) goto cleanup;

    kv->logits = (float *)malloc(vocab_size * sizeof(float));
    if (!kv->logits) goto cleanup;

    kv->final = (float *)malloc(n_embd * sizeof(float));
    if (!kv->final) goto cleanup;

    kv->keys = (uint16_t *)malloc(size_kv_total * sizeof(uint16_t));
    if (!kv->keys) goto cleanup;

    kv->values = (uint16_t *)malloc(size_kv_total * sizeof(uint16_t));
    if (!kv->values) goto cleanup;

    kv->qkv = (float *)malloc(3 * n_embd * sizeof(float));
    if (!kv->qkv) goto cleanup;

    kv->att_scores = (float *)malloc(n_positions * sizeof(float));
    if (!kv->att_scores) goto cleanup;

    kv->attn_out = (float *)malloc(n_embd * sizeof(float));
    if (!kv->attn_out) goto cleanup;

    kv->resid = (float *)malloc(n_embd * sizeof(float));
    if (!kv->resid) goto cleanup;

    kv->ln1 = (float *)malloc(n_embd * sizeof(float));
    if (!kv->ln1) goto cleanup;

    kv->ln2 = (float *)malloc(n_embd * sizeof(float));
    if (!kv->ln2) goto cleanup;

    kv->mlp_hidden = (float *)malloc(4 * n_embd * sizeof(float));
    if (!kv->mlp_hidden) goto cleanup;

    kv->mlp_out = (float *)malloc(n_embd * sizeof(float));
    if (!kv->mlp_out) goto cleanup;

    kv->buf.data = NULL, kv->buf.pos = 0, kv->buf.capacity = 0;

    return 0;
  }

cleanup:
  if (kv) {
    if (kv->x) free(kv->x);
    if (kv->logits) free(kv->logits);
    if (kv->final) free(kv->final);
    if (kv->keys) free(kv->keys);
    if (kv->values) free(kv->values);
    if (kv->qkv) free(kv->qkv);
    if (kv->att_scores) free(kv->att_scores);
    if (kv->attn_out) free(kv->attn_out);
    if (kv->resid) free(kv->resid);
    if (kv->ln1) free(kv->ln1);
    if (kv->ln2) free(kv->ln2);
    if (kv->mlp_hidden) free(kv->mlp_hidden);
    if (kv->mlp_out) free(kv->mlp_out);

    gpt2_float_free(&kv->buf);

    kv->x = kv->logits = kv->final = kv->qkv = kv->att_scores = kv->attn_out = NULL;
    kv->resid = kv->ln1 = kv->ln2 = kv->mlp_hidden = kv->mlp_out = NULL;
    kv->keys = kv->values = NULL;
  }
  return -1;
}

void gpt2_state_free(struct gpt2_model_t *model) {
  if (model) {
    struct gpt2_state_t *kv = &model->state;
    if (kv->x) free(kv->x);
    if (kv->logits) free(kv->logits);
    if (kv->final) free(kv->final);
    if (kv->keys) free(kv->keys);
    if (kv->values) free(kv->values);
    if (kv->qkv) free(kv->qkv);
    if (kv->att_scores) free(kv->att_scores);
    if (kv->attn_out) free(kv->attn_out);
    if (kv->resid) free(kv->resid);
    if (kv->ln1) free(kv->ln1);
    if (kv->ln2) free(kv->ln2);
    if (kv->mlp_hidden) free(kv->mlp_hidden);
    if (kv->mlp_out) free(kv->mlp_out);

    gpt2_float_free(&kv->buf);

    kv->x = kv->logits = kv->final = kv->qkv = kv->att_scores = kv->attn_out = NULL;
    kv->resid = kv->ln1 = kv->ln2 = kv->mlp_hidden = kv->mlp_out = NULL;
    kv->keys = kv->values = NULL;
  }
}

static int gpt2_add(float *out, float *a, float *b, int size) {
  int i;
  if (out == NULL || a == NULL || b == NULL || size <= 0) {
    return -1;
  }
  for(i = 0; i < size; i++) {
    out[i] = a[i] + b[i];
  }
  return 0;
}

static int gpt2_apply_activate(float *x, int size, enum gpt2_activation_t act_type) {
  if (x == NULL || size <= 0) {
      return -1;
  }
  int i;
  for(i = 0; i < size; i++) {
    float xv = x[i];
    switch (act_type) {
      case gpt2_activation_gelu: {
        float cube = 0.044715f * xv * xv * xv;
        float inner = 0.7978845608f * (xv + cube);
        x[i] = 0.5f * xv * (1.0f + tanhf(inner));
        break;
      }
      case gpt2_activation_gelu_new: {
        float inner = 0.7978845608f * (xv + 0.044715f * xv * xv * xv);
        x[i] = 0.5f * xv * (1.0f + tanhf(inner));
        break;
      }
      case gpt2_activation_gelu_fast: {
        x[i] = xv * (0.5f * (1.0f + tanhf(0.797885f * (xv + 0.044715f * xv * xv * xv))));
        break;
      }
      case gpt2_activation_relu: {
        x[i] = (xv > 0.0f) ? xv : 0.0f;
        break;
      }
      case gpt2_activation_silu: {
        x[i] = xv / (1.0f + expf(-xv));
        break;
      }
      case gpt2_activation_tanh: {
        x[i] = tanhf(xv);
        break;
      }
      case gpt2_activation_unknown:
      default:
        return -1;
    }
  }
  return 0;
}

static int gpt2_layernorm(float *out, float *x, float *g, float *b, float eps, int size) {
  int i;
  float mean, val, denom, inv_std;
  if (out == NULL || x == NULL || g == NULL || b == NULL || size <= 0) {
    return -1;
  }
  mean = 0.0f;
  for(i = 0; i < size; i++) mean += x[i];
  mean /= size;
  
  val = 0.0f;
  for(i = 0; i < size; i++) {
    float diff = x[i] - mean;
    val += diff * diff;
  }
  val /= size;
  
  denom = val + eps;
  if (denom <= 0.0f) return -1;
  inv_std = 1.0f / sqrtf(denom);
  for(i = 0; i < size; i++) {
    out[i] = (x[i] - mean) * inv_std * g[i] + b[i];
  }
  return 0;
}

static int gpt2_matmul(float *out, float *x, float *w, float *b, int dim_in, int dim_out) {
  int i, j;
  if (out == NULL || x == NULL || w == NULL || dim_in <= 0 || dim_out <= 0) {
    return -1;
  }
  for (i = 0; i < dim_out; i++) {
    float val = (b != NULL) ? b[i] : 0.0f;
    for (j = 0; j < dim_in; j++) {
      val += x[j] * w[j * dim_out + i];
    }
    out[i] = val;
  }
  return 0;
}

static int gpt2_softmax(float *x, int n) {
  int i;
  float max_val, inv_sum, sum = 0.0f;
  if (x == NULL || n <= 0) {
    return -1;
  }
  max_val = x[0];

  for (i = 1; i < n; i++) {
    if (x[i] > max_val) max_val = x[i];
  }
  for (i = 0; i < n; i++) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  if (sum == 0.0f) return -1;
  inv_sum = 1.0f / sum;
  for (i = 0; i < n; i++) {
    x[i] *= inv_sum;
  }
  return 0;
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

static uint16_t fp32_to_fp16(float f) {
  uint32_t x;
  memcpy(&x, &f, sizeof(x));
  uint32_t sign = (x >> 16) & 0x8000;
  int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mantissa = x & 0x7FFFFF;

  if (exp <= 0) {
    if (exp < -10) {
      return (uint16_t)sign;
    }
    mantissa = (mantissa | 0x800000) >> (1 - exp);
    return (uint16_t)(sign | (mantissa >> 13));
  } else if (exp >= 31) {
    if (exp == 31 && mantissa != 0) {
      return (uint16_t)(sign | 0x7C00 | (mantissa >> 13));
    }
    return (uint16_t)(sign | 0x7C00);
  }
  return (uint16_t)(sign | (exp << 10) | (mantissa >> 13));
}

int gpt2_tensor_to_float(
  struct gpt2_float_t *buf, const struct gpt2_tensor_t *tensor, float **out
) {
  int i, total_elements = 1;

  if (!tensor || !out) return -1;

  for (i = 0; i < tensor->ndim; i++) {
    total_elements *= tensor->shape[i];
  }
  if (tensor->ndim == 0) total_elements = 1;

  if (tensor->dtype == gpt2_dtype_f32) {
    *out = (float *)tensor->data;
    return 0;
  }

  if (!buf || gpt2_float_resize(buf, total_elements) < 0) return -1;

  *out = buf->data + buf->pos;

  switch (tensor->dtype) {
    case gpt2_dtype_i32: {
      int32_t *src = (int32_t *)tensor->data;
      if (!src) return -1;
      for (i = 0; i < total_elements; i++) {
        (*out)[i] = (float)src[i];
      }
      break;
    }
    case gpt2_dtype_i64: {
      int64_t *src = (int64_t *)tensor->data;
      if (!src) return -1;
      for (i = 0; i < total_elements; i++) {
        (*out)[i] = (float)src[i];
      }
      break;
    }
    case gpt2_dtype_f16: {
      uint16_t *src = (uint16_t *)tensor->data;
      if (!src) return -1;
      for (i = 0; i < total_elements; i++) {
        (*out)[i] = fp16_to_fp32(src[i]);
      }
      break;
    }
    case gpt2_dtype_bf16: {
      uint16_t *src = (uint16_t *)tensor->data;
      if (!src) return -1;
      for (i = 0; i < total_elements; i++) {
        (*out)[i] = bf16_to_fp32(src[i]);
      }
      break;
    }
    default: return -1;
  }

  buf->pos += total_elements;;
  return 0;
}

static int gpt2_cache_to_float(
  struct gpt2_float_t *buf, uint16_t *data, float **out, int size
) {
  int i;
  float *dest = NULL;
  if (!data || !out || size <= 0) {
    return -1;
  }
  if (buf) {
    if (gpt2_float_resize(buf, size) != 0) return -1;
    dest = buf->data + buf->pos;
    buf->pos += size;
  } else {
    dest = (float *)malloc(size * sizeof(float));
    if (!dest) return -1;
  }
  *out = dest;
  for (i = 0; i < size; i++) {
    dest[i] = fp16_to_fp32(data[i]);
  }
  return 0;
}

static int gpt2_model_attention(
  struct gpt2_model_t *model, float *out, float *x, 
  float *attn_w, float *attn_b, 
  float *proj_w, float *proj_b, 
  int layer, int pos
) {
  int h, n_embd, head_size, head_stride, layer_stride, base_layer_offset;
  float *q, *k, *v, scale;
  struct gpt2_state_t *state;
  struct gpt2_config_t *config;

  if (!model || !out || !x) return -1;

  state = &model->state;
  config = &model->config;

  n_embd = config->n_embd;
  head_size = config->n_embd / config->n_head;
  
  if (gpt2_matmul(
    state->qkv, x, attn_w, attn_b, n_embd, 3 * n_embd
  ) != 0) {
    return -1;
  }

  q = state->qkv;
  k = state->qkv + n_embd;
  v = state->qkv + 2 * n_embd;

  head_stride = config->n_positions * head_size;
  layer_stride = config->n_head * head_stride;
  base_layer_offset = layer * layer_stride;
  scale = config->scale_attn_weights ? (1.0f / sqrtf((float)head_size)) : 1.0f;

  for (h = 0; h < config->n_head; h++) {
    float *head_q = q + h * head_size;

    int head_base_offset = base_layer_offset + h * head_stride;
    int cache_offset = head_base_offset + pos * head_size;
    
    uint16_t *cache_k = state->keys + cache_offset;
    uint16_t *cache_v = state->values + cache_offset;
    float *scores = state->att_scores;
    float *head_out;
    int t;

    float *src_k = k + h * head_size;
    float *src_v = v + h * head_size;
    for (int i = 0; i < head_size; i++) {
      cache_k[i] = fp32_to_fp16(src_k[i]);
      cache_v[i] = fp32_to_fp16(src_v[i]);
    }

    uint16_t *head_key_cache_base = state->keys + head_base_offset;
    uint16_t *head_val_cache_base = state->values + head_base_offset;

    for (t = 0; t <= pos; t++) {
      uint16_t *past_k_fp16 = head_key_cache_base + t * head_size;
      
      float *past_k = NULL;
      if (gpt2_cache_to_float(
        &state->buf, past_k_fp16, &past_k, head_size
      ) != 0) {
        return -1;
      }

      float score = 0.0f;
      int i;
      for (i = 0; i < head_size; i++) score += head_q[i] * past_k[i];
      scores[t] = score * scale;
    }

    if (gpt2_softmax(scores, pos + 1) != 0) return -1;

    head_out = state->attn_out + h * head_size;
    memset(head_out, 0, head_size * sizeof(float));

    for (t = 0; t <= pos; t++) {
      uint16_t *past_v_fp16 = head_val_cache_base + t * head_size;
      
      float *past_v = NULL;
      if (gpt2_cache_to_float(
        &state->buf, past_v_fp16, &past_v, head_size
      ) != 0) {
        return -1;
      }

      float prob = scores[t];
      int i;
      for (i = 0; i < head_size; i++) head_out[i] += prob * past_v[i];
    }
  }

  return gpt2_matmul(out, state->attn_out, proj_w, proj_b, n_embd, n_embd);
}

static int gpt2_model_transformer_block(
  struct gpt2_model_t *model, float *x, int layer, int pos
) {
  struct gpt2_state_t *s;
  struct gpt2_config_t *config;
  if (!x || !model) return -1;

  s = &model->state;
  config = &model->config;

  int n_embd = config->n_embd;
  float *w_ptr1, *w_ptr2, *w_ptr3, *w_ptr4;
  memcpy(s->resid, x, n_embd * sizeof(float));

  if (gpt2_tensor_to_float(
    &s->buf, &model->layers[layer].ln_1_w, &w_ptr1
  ) != 0) {
    return -1;
  }
  if (gpt2_tensor_to_float(
    &s->buf, &model->layers[layer].ln_1_b, &w_ptr2
  ) != 0) {
    return -1;
  }

  if (gpt2_layernorm(
    s->ln1, x, w_ptr1, w_ptr2, config->layer_norm_epsilon, n_embd
  ) != 0) {
    return -1;
  }
  s->buf.pos = 0;

  if (gpt2_tensor_to_float(
    &s->buf, &model->layers[layer].qkv_w, &w_ptr1
  ) != 0) {
    return -1;
  }
  if (gpt2_tensor_to_float(
    &s->buf, &model->layers[layer].qkv_b, &w_ptr2
  ) != 0) {
    return -1;
  }
  if (gpt2_tensor_to_float(
    &s->buf, &model->layers[layer].attn_proj_w, &w_ptr3
  ) != 0) {
    return -1;
  }
  if (gpt2_tensor_to_float(
    &s->buf, &model->layers[layer].attn_proj_b, &w_ptr4
  ) != 0) {
    return -1;
  }

  if (gpt2_model_attention(
    model, s->attn_out, s->ln1, w_ptr1, w_ptr2, 
    w_ptr3, w_ptr4, layer, pos
  ) != 0) {
    return -1;
  }
  s->buf.pos = 0;
  
  if (gpt2_add(x, s->resid, s->attn_out, n_embd) != 0) return -1;
  memcpy(s->resid, x, n_embd * sizeof(float));

  if (gpt2_tensor_to_float(
    &s->buf, &model->layers[layer].ln_2_w, &w_ptr1
  ) != 0) {
    return -1;
  }
  if (gpt2_tensor_to_float(
    &s->buf, &model->layers[layer].ln_2_b, &w_ptr2
  ) != 0) {
    return -1;
  }
  if (gpt2_layernorm(
    s->ln2, x, w_ptr1, w_ptr2, config->layer_norm_epsilon, n_embd
  ) != 0) {
    return -1;
  }
  s->buf.pos = 0;

  if (gpt2_tensor_to_float(
    &s->buf, &model->layers[layer].fc_w, &w_ptr1
  ) != 0) {
    return -1;
  }
  if (gpt2_tensor_to_float(
    &s->buf, &model->layers[layer].fc_b, &w_ptr2
  ) != 0) {
    return -1;
  }
  if (gpt2_matmul(
    s->mlp_hidden, s->ln2, w_ptr1, w_ptr2, n_embd, 4 * n_embd
  ) != 0) {
    return -1;
  }
  s->buf.pos = 0;
  
  if (gpt2_apply_activate(
    s->mlp_hidden, 4 * n_embd, config->activation_function
  ) != 0) {
    return -1;
  }
  if (gpt2_tensor_to_float(
    &s->buf, &model->layers[layer].proj_w, &w_ptr1
  ) != 0) {
    return -1;
  }
  if (gpt2_tensor_to_float(
    &s->buf, &model->layers[layer].proj_b, &w_ptr2
  ) != 0) {
    return -1;
  }
  if (gpt2_matmul(
    s->mlp_out, s->mlp_hidden, w_ptr1, w_ptr2, 4 * n_embd, n_embd
  ) != 0) {
    return -1;
  }

  return gpt2_add(x, s->resid, s->mlp_out, n_embd);
}

static int gpt2_prob_compare(const void *a, const void *b) {
  const struct gpt2_prob_t *x = (const struct gpt2_prob_t *)a;
  const struct gpt2_prob_t *y = (const struct gpt2_prob_t *)b;
  if (x->val > y->val) return -1;
  if (x->val < y->val) return 1;
  return 0;
}

static int gpt2_model_sample(
  float *logits, int vocab_size, float temperature, int top_k, float top_p,
  float rng, struct gpt2_prob_t *vocab_probs
) {
  if (!logits || vocab_size <= 0 || !vocab_probs) return -1;
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
  if (sum == 0.0f) return -1;
  inv_sum = 1.0f / sum;
  for (i = 0; i < vocab_size; i++) {
    logits[i] *= inv_sum;
  }

  for (i = 0; i < vocab_size; i++) {
    vocab_probs[i].val = logits[i];
    vocab_probs[i].idx = i;
  }

  qsort(vocab_probs, vocab_size, sizeof(struct gpt2_prob_t), gpt2_prob_compare);

  effective_vocab_size = vocab_size;
  if (top_k > 0 && top_k < vocab_size) {
    effective_vocab_size = top_k;
  }

  if (top_p > 0.0f && top_p < 1.0f) {
    float cumulative_prob = 0.0f;
    int last_idx = effective_vocab_size;
    int j;
    for (j = 0; j < effective_vocab_size; j++) {
      cumulative_prob += vocab_probs[j].val;
      if (cumulative_prob > top_p) {
        last_idx = j + 1;
        break;
      }
    }
    effective_vocab_size = last_idx;
  }

  cumulative_sum = 0.0f;
  for (i = 0; i < effective_vocab_size; i++) {
    cumulative_sum += vocab_probs[i].val;
  }
  
  r = rng * cumulative_sum;
  cdf = 0.0f;
  next_token = vocab_probs[0].idx;
  for (i = 0; i < effective_vocab_size; i++) {
    cdf += vocab_probs[i].val;
    if (r < cdf) {
      next_token = vocab_probs[i].idx;
      break;
    }
  }

  return next_token;
}

static float gpt2_random_uniform(unsigned int *seed) {
  *seed = (*seed * 1103515245 + 12345);
  return (float)(*seed & 0x7fffffff) / (float)0x7fffffff;
}

int gpt2_model_generate(
  struct gpt2_model_t *model, struct gpt2_param_t *param,
  int *prompt_tokens, int num_prompt,
  struct gpt2_prob_t *vocab_probs,
  int **tokens, int *token_len
) {
  if (
    !model || !param || !prompt_tokens ||
    !tokens || !token_len || num_prompt <= 0
  ) {
    if (token_len) *token_len = 0;
    return -1;
  }
  struct gpt2_state_t *state = &model->state;
  struct gpt2_config_t *config = &model->config;
  float *wte_ptr, *wpe_ptr, *ln_f_w_ptr, *ln_f_b_ptr, *lm_head_ptr;
  unsigned int fallback_seed = 42;
  int current_token = prompt_tokens[0];
  int pos = 0;
  int capacity = num_prompt + param->max_tokens;
  int count = 0;

  *tokens = (int *)malloc(capacity * sizeof(int));
  if (!(*tokens)) {
    fprintf(stderr, "Error: Memory allocation failed for tokens in generate\n");
    *token_len = 0;
    return -1;
  }

  if (gpt2_tensor_to_float(&state->buf, &model->wte, &wte_ptr) != 0) return -1;
  if (gpt2_tensor_to_float(&state->buf, &model->wpe, &wpe_ptr) != 0) return -1;
  if (gpt2_tensor_to_float(&state->buf, &model->ln_f_w, &ln_f_w_ptr) != 0) return -1;
  if (gpt2_tensor_to_float(&state->buf, &model->ln_f_b, &ln_f_b_ptr) != 0) return -1;
  if (config->tie_word_embeddings) {
    lm_head_ptr = wte_ptr;
  } else {
    if (gpt2_tensor_to_float(&state->buf, &model->lm_head, &lm_head_ptr) != 0) return -1;
  }
  if (!param->seed) {
    param->seed = &fallback_seed;
  }
  while (pos < num_prompt + param->max_tokens) {
    int i;
    int next_token;

    for(i = 0; i < config->n_embd; i++) {
      state->x[i] = wte_ptr[current_token * config->n_embd + i] + wpe_ptr[pos * config->n_embd + i];
    }

    for(i = 0; i < config->n_layer; i++) {
      if (gpt2_model_transformer_block(model, state->x, i, pos) != 0) return -1;
    }

    if (gpt2_layernorm(state->final, state->x, ln_f_w_ptr, ln_f_b_ptr, config->layer_norm_epsilon, config->n_embd) != 0) return -1;

    if (pos < num_prompt - 1) {
      next_token = prompt_tokens[pos + 1];
    } else {
      if (gpt2_matmul(state->logits, state->final, lm_head_ptr, NULL, config->n_embd, config->vocab_size) != 0) return -1;
      float rng = gpt2_random_uniform(param->seed);
      int sample_res = gpt2_model_sample(state->logits, config->vocab_size, param->temperature, param->top_k, param->top_p, rng, vocab_probs);
      if (sample_res < 0) return -1;
      next_token = sample_res;
      printf("Step %d | Token: %d\n", pos, next_token);
      if (next_token == config->eos_token_id) {
        printf("EOS token (%d), stoped.\n", config->eos_token_id);
        if (count < capacity) {
          (*tokens)[count++] = next_token;
        }
        break;
      }
    }

    if (count < capacity) {
      (*tokens)[count++] = next_token;
    }

    pos++;
    current_token = next_token;
    if (pos >= config->n_positions) break;
  }
  *token_len = count;
  return 0;
}

/* Debug */
static const char *get_activation_name(enum gpt2_activation_t act) {
  switch (act) {
    case gpt2_activation_gelu:     return "gelu";
    case gpt2_activation_gelu_new: return "gelu_new";
    case gpt2_activation_relu:     return "relu";
    case gpt2_activation_gelu_fast:     return "gelu_fast";
    case gpt2_activation_silu:     return "silu";
    case gpt2_activation_tanh:     return "tanh";
    default:       return "unknown";
  }
}

static void gpt2_config_dump(const struct gpt2_config_t* config) {
  if (config == NULL) {
    printf("Config pointer is NULL!\n");
    return;
  }

  printf("=========================================\n");
  printf("           GPT-2 CONFIGURATION           \n");
  printf("=========================================\n");

  printf(" [Architecture]\n");
  printf("   - Vocab Size         : %d\n", config->vocab_size);
  printf("   - Max Positions (Context): %d\n", config->n_positions);
  printf("   - Embedding Dim (n_embd) : %d\n", config->n_embd);
  printf("   - Layers (n_layer)   : %d\n", config->n_layer);
  printf("   - Heads (n_head)     : %d\n", config->n_head);

  printf("\n [Hyperparameters]\n");
  printf("   - LayerNorm Epsilon  : %e\n", config->layer_norm_epsilon);
  printf(
    "   - Activation Func    : %s\n",
    get_activation_name(config->activation_function)
  );
  printf(
    "   - Scale Attn Weights : %s\n",
    config->scale_attn_weights ? "True" : "False"
  );
  printf(
    "   - Tie Word Embeddings: %s\n",
    config->tie_word_embeddings ? "True" : "False"
  );

  printf("\n [Special Token IDs]\n");
  printf("   - BOS Token ID       : %d\n", config->bos_token_id);
  printf("   - EOS Token ID       : %d\n", config->eos_token_id);
  printf("   - PAD Token ID       : %d\n", config->pad_token_id);
  printf("   - UNK Token ID       : %d\n", config->unk_token_id);
  printf("=========================================\n");
}

static void print_gpt2_string(struct gpt2_string_t gpt2) {
  if (gpt2.buf == NULL || gpt2.len == 0) return;
  size_t i = 0;
  while (i < gpt2.len) {
    if (
      i + 1 < gpt2.len && (unsigned char)gpt2.buf[i] == 0xC4 &&
      (unsigned char)gpt2.buf[i+1] == 0xA0
    ) {
      putchar(' ');
      i += 2;
    } else {
      putchar(gpt2.buf[i]);
      i++;
    }
  }
}

static void gpt2_tokenize_dump(
  const struct gpt2_tokenize_t *tokenize, int max_items
) {
  if (!tokenize) {
    printf("Tokenize struct: NULL\n");
    return;
  }
  size_t vocab_size = tokenize->tokens.size;
  size_t merges_size = tokenize->merges.size;
  printf("==================================================\n");
  printf("               GPT-2 TOKENIZER INFO               \n");
  printf("==================================================\n");
  printf(
    "Vocab Size      : %zu (Capacity: %zu)\n",
    vocab_size, tokenize->tokens.capacity
  );
  printf(
    "Merges Size     : %zu (Capacity: %zu)\n",
    merges_size, tokenize->merges.capacity
  );
  printf("--------------------------------------------------\n");
  printf("\n[VOCAB TOKENS]\n");
  size_t printed_vocab = 0;
  size_t print_vocab_limit = vocab_size;
  if (max_items > 0 && (size_t)max_items < vocab_size) {
    print_vocab_limit = (size_t)max_items;
  }

  for (
    size_t i = 0;
    i < tokenize->tokens.capacity && printed_vocab < print_vocab_limit;
    i++
  ) {
    struct gpt2_node_t t = tokenize->tokens.buckets[i];
    if (t.state != gpt2_cell_empty) {
      printf("  [%5zu] ID: %-6d | Token: \"", printed_vocab, t.value);
      print_gpt2_string(t.key);
      printf("\" (len: %zu)\n", t.key.len);
      printed_vocab++;
    }
  }

  if (printed_vocab < vocab_size) {
    printf(
      "  ... và %zu token khác chưa được in.\n",
      vocab_size - printed_vocab
    );
  }

  /* --- In danh sách Merges --- */
  printf("\n[MERGE RULES]\n");
  size_t printed_merges = 0;
  size_t print_merges_limit = merges_size;
  if (max_items > 0 && (size_t)max_items < merges_size) {
    print_merges_limit = (size_t)max_items;
  }

  for (
    size_t i = 0;
    i < tokenize->merges.capacity && printed_merges < print_merges_limit;
    i++
  ) {
    struct gpt2_node_t m = tokenize->merges.buckets[i];
    if (m.state != gpt2_cell_empty) {
      printf("  [%5zu] Rank: %-5d | Merge: (\"", printed_merges, m.value);
      print_gpt2_string(m.key);
      printf("\")\n");
      printed_merges++;
    }
  }

  if (printed_merges < merges_size) {
    printf(
      "  ... và %zu quy tắc merge khác chưa được in.\n",
      merges_size - printed_merges
    );
  }

  printf("==================================================\n");
}
/* Debug */

int main(int argc, char **argv) {
  if (argc > 1) {
    struct gpt2_model_t model = {0};
    int num_prompt, *prompt_tokens = NULL;
    unsigned int seed = 42;
    // char *respond = NULL;
    const char *text = "Việt Nam là quốc gia có";
    struct gpt2_param_t param = {
      50, 40, 0.8f, 0, &seed
    };
    if (gpt2_load_model(argv[1], &model) != GPT2_OK) {
      return -1;
    }
    gpt2_config_dump(&model.config);
    gpt2_tokenize_dump(&model.tokenize, 5);
    if (gpt2_state_init(&model)) {
      gpt2_model_free(&model);
      return -1;
    }
    if (argc > 2) {
      text = argv[2];
    }
    printf("Input:\n %s\n", text);
    num_prompt = gpt2_model_encode(&model, text, &prompt_tokens);
    if (num_prompt > 0) {
      struct gpt2_prob_t *vocab_probs = (struct gpt2_prob_t *)malloc(model.config.vocab_size * sizeof(struct gpt2_prob_t));
      if (vocab_probs) {
        int *tokens = NULL, ntok = 0;
        if (gpt2_model_generate(&model, &param, prompt_tokens, num_prompt, vocab_probs, &tokens, &ntok) == 0) {
          char *decoded = NULL;
          if (gpt2_model_decode(&model, tokens, ntok, &decoded) > 0) {
            printf("Decoded:\n %s\n", decoded);
            free(decoded);
          }
          free(tokens);
        }
        free(vocab_probs);
      }
      free(prompt_tokens);
    }
    /*if (gpt2_model_generate(&model, &param, text, &respond) != -1) {
      printf("Respond: %s\n", respond);
      free(respond);
    }*/
    // gpt2_model_generate(&model, text, 50, 0.8f, 40, 0.9, &seed);
    gpt2_state_free(&model);
    gpt2_model_free(&model);
    return GPT2_OK;
  }
  return -1;
}
