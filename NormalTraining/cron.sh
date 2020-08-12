#!/bin/bash
currenttime=$(date +%H:%M)
check=$(ps cax| grep python sign_api.py)
if [ "$check" == "" ];then
	nohup python sign_api.py
fi

