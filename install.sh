#!/bin/bash

# Install the dependencies
if [ command -v sex &> /dev/null ]; then
    echo SExtractor not found. Installing...
    echo Getting SExtractor from http://www.astromatic.net/software/sextractor
    git clone https://github.com/astromatic/sextractor.git $HOME/.local/sextractor
    cd $HOME/.local/sextractor
    # if [ command -v cfitsio &> /dev/null ]; then
    #     echo cfitsio not found. Installing...
    #     echo Getting cfitsio from https://heasarc.gsfc.nasa.gov/fitsio/
    #     git clone 
    ./autogen.sh
    if [ "$?" -ne "0" ]; then
        echo "Error: autogen.sh failed"
        exit 1
    fi
    ./configure --prefix=$HOME/.local
else
    echo SExtractor is already installed
fi
