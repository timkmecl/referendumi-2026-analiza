#!/usr/bin/env bash
# Požene celotno analizo od surovih podatkov do slik. Zaženi iz korenske mape paketa:  bash run_all.sh
set -e
cd "$(dirname "$0")"
mkdir -p rezultati slike izpisi
for f in koda/[0-9]*.py koda/s*.py; do
  n=$(basename "$f" .py); echo ">>> $n"
  python3 "$f" > "izpisi/$n.txt" 2>&1 || { echo "NAPAKA v $f (glej izpisi/$n.txt)"; exit 1; }
done
echo "Končano. Številke so v mapi izpisi/, slike v slike/, vmesni rezultati v rezultati/."
