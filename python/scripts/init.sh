#!/bin/bash
cd $(dirname ${BASH_SOURCE[0]})

apt-get install 7zip-full

cd ../cq_data
rm -rf cq
7za x cq.7z -ocq -y
