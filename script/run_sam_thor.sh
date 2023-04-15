#!/usr/bin/env bash

python model/samprompt.py --checkpoint sam_vit_h_4b8939.pth --model-type vit_h --output docs/mask --input docs/thor/d_second.png --px 440 --py 275