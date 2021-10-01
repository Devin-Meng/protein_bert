#!/bin/bash

ps -aux | grep Zhaoxu | grep  "python" | cut -c 9-15 |xargs kill -9
