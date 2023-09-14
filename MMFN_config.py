MMFN_config ={
    "BERT_size" : 768,

    # clip output = batch *  clip_size
    "CLIP_size" : 512,

    # swim output last_hidden_state =  1 * 144 * 1024
    "SWIN_max_length" : 144,
    "SWIN_size" : 1024,

    # xlnet output = batch * xlnet_max_length * xlnet_size
    "XLNET_size" : 768,
    "xlent_max_length" : 144,
    "bert_max_length" : 144,

    "batch_size" : 32,

    "d_model" : 512,
    "k_dim" : 256,
    "v_dim" : 256 ,
    "num_heads" : 16,
}