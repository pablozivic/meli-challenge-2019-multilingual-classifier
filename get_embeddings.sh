#!/bin/sh

[ ! -d "resources" ] && mkdir resources

wget https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.pt.vec --directory resources
wget https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.es.vec --directory resources
wget https://dl.fbaipublicfiles.com/arrival/vectors/wiki.multi.en.vec --directory resources