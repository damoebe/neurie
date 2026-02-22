package me.damoebe.transformer.mha;

import me.damoebe.mlp.structure.Connection;
import me.damoebe.transformer.Embedding;

import java.util.ArrayList;
import java.util.List;

/**
 * The abstract Head class, every Head type must extend.
 */
public abstract class Head {
    /**
     * All QKV weights stored in one list.
     */
    protected List<Double> weights = new ArrayList<>();
    protected List<Double> biases = new ArrayList<>();

    protected final int inputEmbeddingSize;
    protected final int inputEmbeddingAmount;

    /**
     * The main constructor of a Head
     * @param inputEmbeddingSize The size of each input embedding
     * @param inputEmbeddingAmount The amount of input embeddings
     */
    protected Head(int inputEmbeddingAmount, int inputEmbeddingSize){
        this.inputEmbeddingAmount = inputEmbeddingAmount;
        this.inputEmbeddingSize = inputEmbeddingSize;
        for (int i = 0; i != inputEmbeddingAmount*inputEmbeddingSize*3; i++){
            weights.add(Connection.getRandomWeight());
            biases.add(Math.random() * 2 - 1);
        }
    }

    /**
     * Generates an output of this head by using the input and head-weights to generate query, key and value, which
     * get merged to a new Embedding List containing the output.
     * @param inputEmbeddings The input Embedding list
     * @return The output for the input Embeddings as an Embedding list.
     * @throws Exception If the input does not fit in this head. see isValidInput().
     */
    public abstract List<Embedding> getOutputFor(List<Embedding> inputEmbeddings) throws Exception;

    /**
     * Checks if a List of Embeddings matches the objects inputEmbeddingSize and inputEmbeddingAmount attribute values
     * @param embeddings A list of the (input) embeddings that should be checked
     * @return true if the Input matches the requirements, false if the input is invalid for this head
     */
    protected boolean isValidInput(List<Embedding> embeddings){
        if (embeddings.size() != inputEmbeddingAmount) return false;
        for (Embedding embedding : embeddings){
            if (embedding.getEmbeddingSize() != inputEmbeddingSize) return false;
        }
        return true;
    }
}
