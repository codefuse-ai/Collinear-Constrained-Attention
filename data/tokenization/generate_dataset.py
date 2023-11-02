# origin: 使用GPT-Neox原生的Encoder：key为text，只生成input_ids，训练时document首尾相连按窗口去取
# Prompt_padding：使用UniformEncoder：key为input_ids和loss_mask; loss_mask保证只训练Target部分的Loss；每条样本Padding到seq_length，避免了一个Sample里包含多个样本的问题，但缺点是比较浪费计算资源
prompt_padding_cmd = "python preprocess_data.py \
            --input {input} \
            --jsonl-keys input_ids loss_mask \
            --output-prefix {output_prefix} \
            --vocab ../../tokenizer-ant-v3.json  \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --workers {worker} \
            --encoder UniformEncoder \
            --seq-length {seq_length} \
            --mode sft \
            --padding"

align_padding_cmd = "python preprocess_data_align.py \
            --input {input} \
            --jsonl-keys w_input_ids w_loss_mask l_input_ids l_loss_mask \
            --output-prefix {output_prefix} \
            --vocab ../../tokenizer-ant-v3.json  \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --workers {worker} \
            --encoder UniformEncoder \
            --seq-length {seq_length} \
            --mode align \
            --padding"

origin_cmd = "python preprocess_data.py \
            --input {input} \
            --output-prefix {output_prefix} \
            --vocab ../../tokenizer-ant-v3.json  \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --workers {worker} \
            --encoder OriginEncoder \
            --append-eod"

convert_dict = {
    "origin":{
        "output_path":"xxx",
        "cmd": origin_cmd
    },
    "prompt_padding":{
        "output_path":"/ossfs/workspace/coh_tokenization",
        "cmd": prompt_padding_cmd
    },
    "align_padding":{
        "output_path":"/ossfs/workspace/alignment_tokenization",
        "cmd": align_padding_cmd
    }
}

input_dict = {
    'dataset1': "/path/dataset1_path",
    'dataset2': "/path/dataset2_path"
}
# conver_type_list = ["align_padding"]
conver_type_list = ["prompt_padding"]
output_name = "coh"

if __name__ == "__main__":
    import os
    seq_length = 2048
    worker = 16

    for convert_type in conver_type_list:
        convert_info = convert_dict[convert_type]
        output_path = convert_info["output_path"]
        convert_cmd = convert_info["cmd"]
        output_prefix = os.path.join(output_path, output_name)

        input_ = ",".join(input_dict.values())
        print(input_)
        cmd = convert_cmd.replace("{input}", input_).replace("{output_prefix}", output_prefix).replace("{seq_length}", str(seq_length)).replace("{worker}", str(worker))
        os.system(cmd)