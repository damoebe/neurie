package me.damoebe.transformer.embedding;

import java.util.ArrayList;
import java.util.List;

/**
 * A container class for a transformer internal Embedding-Sequence
 */
public class Sequence {
    private final List<Embedding> embeddings;

    /**
     * Creates a new empty Embedding-Sequence.
     */
    public Sequence(){
        this.embeddings = new ArrayList<>();
    }

    /**
     * Constructs a Sequence from a raw-input-data list by tokenizing it and positional-encode each token.
     * @param rawSequenceData The input sequence (un-tokenized) as a Double List.
     * @param tokenSize The data-size each token (embedding) should have.
     */
    public Sequence(List<Double> rawSequenceData, int tokenSize){
        this(); // create empty list
        if (rawSequenceData.isEmpty() || rawSequenceData.size()%tokenSize != 0)
            throw new RuntimeException("The input can not be tokenized because of invalid parameters");
        int i = 0;
        for (int embeddingIndex = 0; embeddingIndex != rawSequenceData.size()/tokenSize; embeddingIndex++){
            List<Double> embeddingData = new ArrayList<>();
            for (int embeddingValue = 0; embeddingValue != tokenSize; embeddingValue++){
                embeddingData.add(rawSequenceData.get(i));
                i++;
            }
            Embedding embedding = new Embedding();
            this.embeddings.add(embedding.fromToken(new Token(embeddingData), embeddingIndex));
        }
    }

    /**
     * Getter for the stored embeddings
     * @return A list with all embeddings
     */
    public List<Embedding> embeddings(){
        return embeddings;
    }
}
