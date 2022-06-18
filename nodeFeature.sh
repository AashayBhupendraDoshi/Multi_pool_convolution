# Here we will perform experiments based on the length (number of blocks)
# and breath (number of blocks) of the network architecture.
# Since embeddings (emb) must be a function of number of heads (h)
# We test test for two embedding sizes, 64 and 128, and accordingly
# test for different head sizes.
alpha=0
for dt in "proteins_full" "dd" "frankenstein" "enzymes"
do
    for emb in 128
    do
        for h in 4
        do
            for b in 2
            do
                python3 ./GraphClassification_nodeFeature.py --h=$h --b=$b --emb=$emb --dt=$dt --alpha=$alpha
            done
        done

    done
done

#chmod +x ./noFeature.sh
#./noFeature.sh