#!/bin/sh
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 srcFolder dstFolder" >&2
  exit 1
fi
srcFolder=$1
dstFolder=$2
imgExtensionList="jpg jpeg JPG png PNG"
for file in "$srcFolder"/*
do
  filename=$(basename -- "$file")
  extension="${filename##*.}"
  for imgExtension in $imgExtensionList
  do
    if [ "$extension" == "$imgExtension" ]; then
      dstFile="${dstFolder}/${filename}"
      if [ ! -e "$dstFile" ]; then
        ln -s "$file" "$dstFile"
      fi
      break
    fi
  done
done