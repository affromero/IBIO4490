#!/bin/sh

echo NOT AVAILABLE > errors.txt
lab="01"

while read u
do
    REPO=https://github.com/$u/IBIO4490
    #test connection
    echo $REPO
    wget -q --spider $REPO
    status=$?
    if [ $status -eq 0 ]
    then
        folder_lab=lab$lab/$u
        rm -rf $folder_lab
        git clone --progress https://github.com/$u/IBIO4490.git $folder_lab
    else
        echo Not available
        echo $u >> errors.txt
    fi
    sleep 10
    echo ""
done < links_ibio4490.txt
