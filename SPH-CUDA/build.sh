#!/bin/bash
#echo building the project

cmake . -B./build/
make -C ./build/ --jobs=8

