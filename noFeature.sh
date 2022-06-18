# Here we will perform experiments based on the length (number of blocks)
# and breath (number of blocks) of the network architecture.
# Since embeddings (emb) must be a function of number of heads (h)
# We test test for two embedding sizes, 64 and 128, and accordingly
# test for different head sizes.
alpha=0
#"twitch_egos" "reddit_threads" "twitter-real-graph-partial"
for dt in "imdb-binary" "imdb-multi" "reddit-binary" "collab"  "reddit-multi-5k" "reddit-multi-12k"
do
    for alpha in 0
    do
        for emb in 32
        do
            for b in 4
            do
                for h in 2
                do
                    python3 ./GraphClassification_noFeature.py --h=$h --b=$b --emb=$emb --dt=$dt --alpha=$alpha
                done
            done

        done
    done
done