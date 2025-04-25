#!/bin/sh

# folositor pentru WSL
jupyter-lab --no-browser --ip `ip addr | grep eth0 | grep inet | awk '{print $2}' | cut -d"/" -f1`
