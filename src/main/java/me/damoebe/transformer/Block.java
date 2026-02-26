package me.damoebe.transformer;

import me.damoebe.mlp.Network;
import me.damoebe.transformer.mha.EDHead;
import me.damoebe.transformer.mha.Head;
import me.damoebe.transformer.mha.MultiHeadAttention;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.List;

/**
 * The Transformer Bock class containing a mha and multiple mlps.
 * @param <H> The Head type that should be used to construct the MultiHeadAttention attribute.
 * @param <N> The MLP type that will be used in this block.
 */
public class Block <H extends Head, N extends Network>{

    private final Normalization preMHANorm;
    private final MultiHeadAttention<H> multiHeadAttention;
    private final Normalization preMLPNorm;
    private final List<N> multiLayerPerceptrons;

    /**
     * The main constructor of the Block class
     * @param multiHeadAttention A MultiHeadAttention object that will be cloned and used for this block.
     * @param multiLayerPerceptron A Network object that will be cloned and used as an MLP architecture.
     */
    public Block(@NotNull MultiHeadAttention<H> multiHeadAttention, @NotNull N multiLayerPerceptron){
        this.multiHeadAttention = multiHeadAttention.clone();
        List<N> mlps = new ArrayList<>();

        for (int embedding = 0; embedding != multiHeadAttention.getEmbeddingAmount(); embedding++){
            @SuppressWarnings("unchecked") N clonedMLP = (N) multiLayerPerceptron.clone();
            mlps.add(clonedMLP);
        }

        if (multiHeadAttention.getHeads().getFirst() instanceof EDHead){
            // times two because ED Heads have twice as many inputs
            this.preMHANorm = new Normalization(multiHeadAttention.getEmbeddingAmount(), multiHeadAttention.getEmbeddingSize()*2);
        }else {
            this.preMHANorm = new Normalization(multiHeadAttention.getEmbeddingAmount(), multiHeadAttention.getEmbeddingSize());
        }

        this.preMLPNorm = new Normalization(multiHeadAttention.getEmbeddingAmount(), multiHeadAttention.getEmbeddingSize());
        this.multiLayerPerceptrons = mlps;
    }

    /**
     * Gets the output for the inserted input(s) by passing them through the mha module and mlp.
     * @param inputEmbeddings A list of inputs: if the head type is EDHead: [0] is the decoder input and [1] is the encoder input.
     *                        Else: [0] is the head input.
     * @return The output embeddings for this block.
     */
    public Sequence getOutputFor(Sequence... inputEmbeddings){
        Sequence mhaOutput = multiHeadAttention.getOutputFor(preMHANorm.normalize(inputEmbeddings));
        Sequence mlpOutput = new Sequence();
        int i = 0;
        for (Embedding embedding : preMLPNorm.normalize(mhaOutput)[0].embeddings()){
            multiLayerPerceptrons.get(i).insertInput(embedding.data());
            multiLayerPerceptrons.get(i).updateAllActivations();
            mlpOutput.embeddings().add(new Embedding(multiLayerPerceptrons.get(i).getOutput()));
            i++;
        }
        return mlpOutput;
    }
}
