#!/bin/bash
cd $(dirname ${BASH_SOURCE[0]})

$1 freeze_graph.$2.py --input_meta_graph train/ckpt-$3.meta --input_binary True --input_checkpoint train/ckpt-$3 --output_node_names Generator_variables/gen_image --output_graph output.pb
