package me.damoebe.transformer;

import java.util.ArrayList;
import java.util.List;

/**
 * A record class for an internal Embedding-Sequence
 * @param embeddings The embeddings stored in this sequence
 */
public record Sequence(List<Embedding> embeddings) {
    /**
     * Creates a new empty Embedding-Sequence.
     */
    public Sequence(){
        this(new ArrayList<>());
    }

}
