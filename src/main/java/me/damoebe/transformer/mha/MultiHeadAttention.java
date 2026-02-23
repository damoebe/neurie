package me.damoebe.transformer.mha;

import me.damoebe.mlp.structure.Connection;

import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.List;

/**
 * A container class for Heads
 * @param <H> The type of head that is used stored in the class
 */
public class MultiHeadAttention<H extends Head> {
    /**
     * A list of all heads that are contained in a MultiHeadAttention object
     */
    private List<H> heads = new ArrayList<>();

    /**
     * A weight matrix that is used to convert the concat-heads-matrix into a matrix with the dimensions of the
     * input embedding-list
     */
    private double[][] weights;

    /**
     * Main constructor for the MultiHeadAttention class
     * @param c The class that is being used for the heads
     * @param headAmount The amount of heads this MHA object should have
     * @param inputEmbeddingAmounts The amount of the input embeddings
     * @param inputEmbeddingSize The size of the input embeddings
     */
    public MultiHeadAttention(Class<H> c, int headAmount, int inputEmbeddingAmounts, int inputEmbeddingSize)  {
        // initialize heads
        try {
            Constructor<H> headConstructor = c.getConstructor(Integer.class, Integer.class);
            for (int h = 0; h != headAmount; h++) {
                H head;
                head = headConstructor.newInstance(inputEmbeddingAmounts, inputEmbeddingSize);
                heads.add(head);
            }
        }catch (Exception e){
            throw new RuntimeException();
        }
        // initialize weights
        for (int row = 0; row != heads.getFirst().inputEmbeddingAmount; row++){
            for (int column = 0; column != heads.getFirst().inputEmbeddingSize; column++){
                weights[row][column] = Connection.getRandomWeight();
            }
        }
    }
    //TODO: get output for input method feat. matrix concat method + using matrix multiply method from Head for weight matrix
}
