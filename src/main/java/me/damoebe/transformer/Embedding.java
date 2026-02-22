package me.damoebe.transformer;

import java.util.List;

/**
 * Record class for Transformer intern data.
 * @param data The data this Embedding contains as a Double List.
 */
public record Embedding(List<Double> data) {

    /**
     * Getter for the size of the data array.
     * @return The data Double List size.
     */
    public int getEmbeddingSize(){
        return data.size();
    }
}
