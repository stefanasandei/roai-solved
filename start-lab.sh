#!/bin/sh

if [[ $(grep -i Microsoft /proc/version) ]]; then
    # folositor pentru WSL
    jupyter-lab --no-browser --ip `ip addr | grep eth0 | grep inet | awk '{print $2}' | cut -d"/" -f1`
else
    # GNU/Linux normal
    jupyter-lab --no-browser
fi
