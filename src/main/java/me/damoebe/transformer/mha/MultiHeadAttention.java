package me.damoebe.transformer.mha;

import me.damoebe.mlp.structure.Connection;
import me.damoebe.transformer.Embedding;

import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

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
    public MultiHeadAttention(Class<H> c, int headAmount, int inputEmbeddingAmounts, int inputEmbeddingSize, boolean masked)  {
        // initialize heads
        try {
            Constructor<H> headConstructor = c.getConstructor(int.class, int.class, boolean.class);
            for (int h = 0; h != headAmount; h++) {
                H head;
                head = headConstructor.newInstance(inputEmbeddingAmounts, inputEmbeddingSize, masked);
                this.heads.add(head);
            }
        }catch (Exception e){
            throw new RuntimeException("H typed Class could not be loaded: " + e.getMessage());
        }

        // initialize weights
        this.weights = new double[heads.getFirst().inputEmbeddingAmount][heads.getFirst().inputEmbeddingSize];
        for (int row = 0; row != heads.getFirst().inputEmbeddingAmount; row++){
            for (int column = 0; column != heads.getFirst().inputEmbeddingSize; column++){
                weights[row][column] = Connection.getRandomWeight();
            }
        }
    }

    /**
     * Generates an output for a list of inputs(embedding lists).
     * @param input An input as a list of embedding lists -> see EDHead doc if you are using a list size larger than 1
     * @return The output embedding list
     */
    public List<Embedding> getOutputFor(List<Embedding>[] input){
        List<double[][]> matrices = new ArrayList<>();
        for (H head : heads){
            try {
                head.insertInput(input);
            }catch (Exception e){
                throw new RuntimeException(e.getMessage());
            }
            matrices.add(head.getOutput());
        }
        List<Embedding> resultEmbeddings = new ArrayList<>();
        double[][] outputMatrix = Head.multiplyMatrices(concatMatrices(matrices), weights);
        for (int row = 0; row != Objects.requireNonNull(outputMatrix).length; row++){
            List<Double> embeddingData = new ArrayList<>();
            for (int column = 0; column != outputMatrix[0].length; column++){
                embeddingData.add(outputMatrix[row][column]);
            }
            resultEmbeddings.add(new Embedding(embeddingData));
        }
        return resultEmbeddings;
    }

    /**
     * Merges a list of matrices.
     * @param matrices the matrices that will be concat
     * @return the merged matrix as a 2d array
     */
    public static double[][] concatMatrices(List<double[][]> matrices){
        int columns = 0;
        int rows = matrices.getFirst().length;
        for (double[][] matrix : matrices){
            columns += matrix[0].length;
        }
        double[][] resultMatrix = new double[rows][columns];
        int total_row = 0;
        for (double[][] matrix : matrices){
            for(int row = 0; row != matrix.length; row++){
                System.arraycopy(matrix[row], 0, resultMatrix[total_row], 0, matrix[0].length);
            }
        }
        return resultMatrix;
    }

    public MultiHeadAttention<H> clone(){
        try {
            //noinspection unchecked
            return (MultiHeadAttention<H>) super.clone();
        }catch (Exception e){
            throw new RuntimeException(e.getMessage());
        }
    }

    public int getEmbeddingAmount(){
        return this.heads.getFirst().inputEmbeddingAmount;
    }

    public int getEmbeddingSize(){
        return this.heads.getFirst().inputEmbeddingSize;
    }

    public List<H> getHeads(){
        return this.heads;
    }
}
