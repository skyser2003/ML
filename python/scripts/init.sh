#!/bin/bash
cd $(dirname ${BASH_SOURCE[0]})

cd ../cq_data
rm -rf cq
7za x cq.7z -ocq -y
