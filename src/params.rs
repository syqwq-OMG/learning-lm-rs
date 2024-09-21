use std::fmt::format;
use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::{SafeTensors, View};
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // todo!("实现从safetensors文件的模型参数加载");
        let get_tensor = |name: &str| {
            let tensor_view = safetensor.tensor(name).unwrap();
            let shape = tensor_view.shape();
            let data = tensor_view.data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            Tensor::new(data, &shape.to_vec())
        };

        let get_tensor_vec = |name: &str| {
            let layers=config.num_hidden_layers;
            (0..layers)
                .map(|layer| get_tensor(&format!("model.layers.{layer}.{name}")))
                .collect::<Vec<Tensor<f32>>>()
        };

        LLamaParams {
            embedding_table: get_tensor("lm_head.weight"),
            rms_out_w: get_tensor("model.norm.weight"),    // 维度是 (1, 128)
            lm_head: get_tensor("lm_head.weight"), // 和 embedding_table 可以相同, 取决于 tie_word_embeddings 配置项

            // 结构为 Vec<Tensor<f32>>
            rms_att_w: get_tensor_vec("input_layernorm.weight"),
            rms_ffn_w: get_tensor_vec("post_attention_layernorm.weight"),
            wq: get_tensor_vec("self_attn.q_proj.weight"),
            wk: get_tensor_vec("self_attn.k_proj.weight"),
            wv: get_tensor_vec("self_attn.v_proj.weight"),
            wo: get_tensor_vec("self_attn.o_proj.weight"),
            w_up: get_tensor_vec("mlp.up_proj.weight"),
            w_gate: get_tensor_vec("mlp.gate_proj.weight"),
            w_down: get_tensor_vec("mlp.down_proj.weight"),

        }
    }
}
