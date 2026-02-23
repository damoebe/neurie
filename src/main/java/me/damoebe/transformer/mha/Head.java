package me.damoebe.transformer.mha;

import me.damoebe.mlp.structure.Connection;
import me.damoebe.transformer.Embedding;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * The Decoder and Encoder Head Object class
 */
public class Head {

    /**
     * All weights of the head in one list, (size = inputEmbeddingSize*inputEmbeddingAmount*3) -> used to
     * calculate the queries, keys and values.
     */
    protected List<Double> weights = new ArrayList<>();

    /**
     * All biases of the head in one list, (size = inputEmbeddingSize*inputEmbeddingAmount*3) -> used to
     * calculate the queries, keys and values.
     */
    protected List<Double> biases = new ArrayList<>();

    /**
     * The current attention matrix (2d array) -> softmax(dot product/scale + mask)
     */
    private double[][] attention;
    /**
     * The current value matrix. Rows are embeddings, columns are embedding vectors
     */
    private double[][] values;

    /**
     * The vector dimension (size) that each inserted embedding must have
     */
    protected final int inputEmbeddingSize;
    /**
     * The amount of embeddings that are allowed to be inserted.
     */
    protected final int inputEmbeddingAmount;

    /**
     * The main constructor of the decoder Head
     * Identical to the super constructor
     * @param inputEmbeddingAmount The amount of input embeddings
     * @param inputEmbeddingSize The size of each input embedding
     */
    public Head(int inputEmbeddingAmount, int inputEmbeddingSize) {
        this.inputEmbeddingAmount = inputEmbeddingAmount;
        this.inputEmbeddingSize = inputEmbeddingSize;

        this.attention = new double[inputEmbeddingAmount][inputEmbeddingAmount];
        this.values = new double[inputEmbeddingAmount][inputEmbeddingSize];

        for (int i = 0; i != inputEmbeddingAmount*inputEmbeddingSize*3; i++){
            weights.add(Connection.getRandomWeight());
            biases.add(Math.random() * 2 - 1);
        }
    }

    /**
     * Generates an output for one input embedding list, regarding the decoder head attention rules.
     * @param inputEmbeddingLists The input Embedding lists -> here it can only contain one embedding-list
     * @throws Exception if the inserted input embeddings don't fit the expectations
     */
    public void insertInput(List<Embedding>[] inputEmbeddingLists) throws Exception{
        if (inputEmbeddingLists.length != 1) throw new Exception("This type of head can only take one Embedding list.");

        if (!isValidInput(inputEmbeddingLists[0])) throw new Exception("Not a valid input Embedding list!");

        List<List<double[]>> QKVMatrices = getQKVMatrices(inputEmbeddingLists);

        List<double[]> queries = QKVMatrices.getFirst();
        List<double[]> keys = QKVMatrices.get(1);
        List<double[]> values = QKVMatrices.getLast();

        this.values = values.toArray(new double[values.size()][]);

        double[][] scaled_masked_dot_product = new double[queries.size()][keys.size()];
        for (int q = 0; q != queries.size(); q++){
            for (int k = 0; k != keys.size(); k++){
                // triangular mask (causal mask)
                if (k > q){
                    scaled_masked_dot_product[q][k] = -Double.MAX_VALUE;
                }

                // calculate dot product (sum(vec*vec))
                double sum = 0;
                double[] query = queries.get(q);
                double[] key = keys.get(k);
                for (int i = 0; i != query.length; i++){
                    sum *= query[i] * key[i];
                }
                scaled_masked_dot_product[q][k] = sum / Math.sqrt(key.length); // scaling
            }
        }

        double[][] attention = new double[queries.size()][keys.size()];
        for (int row = 0; row != queries.size(); row++){
            attention[row] = softmax(scaled_masked_dot_product[row]);
        }
        this.attention = attention;
    }

    /**
     * Generates a list of query, key and value matrices based on the heads weights and biases.
     * @param inputEmbeddingLists The input Embedding lists -> here it can only contain one embedding-list
     * @return a list of query, key and value matrices
     */
    protected List<List<double[]>> getQKVMatrices(List<Embedding>[] inputEmbeddingLists){
        List<List<double[]>> QKV = new ArrayList<>();
        int weightIndex = 0;

        List<double[]> queries = new ArrayList<>();
        List<double[]> keys = new ArrayList<>();
        List<double[]> values = new ArrayList<>();

        for (Embedding inputEmbedding : inputEmbeddingLists[0]){
            double[] query = new double[this.inputEmbeddingSize];
            double[] key = new double[this.inputEmbeddingSize];
            double[] value = new double[this.inputEmbeddingSize];
            int i = 0;
            for (Double embeddingValue : inputEmbedding.data()){
                query[i] = (embeddingValue * this.weights.get(weightIndex) + this.biases.get(weightIndex));
                weightIndex++;
                key[i] = (embeddingValue * this.weights.get(weightIndex) + this.biases.get(weightIndex));
                weightIndex++;
                value[i] = (embeddingValue * this.weights.get(weightIndex) + this.biases.get(weightIndex));
                weightIndex++;
                i++;
            }
            queries.add(query);
            keys.add(key);
            values.add(value);
        }
        QKV.add(queries);
        QKV.add(keys);
        QKV.add(values);

        return QKV;
    }

    /**
     * Generates an output of this head by using the current attention, which is multiplied with the current value-vectors
     * @return The output for the input Embeddings as an Embedding list.
     */
    public List<Embedding> getOutput(){
        // TODO: multiply attention matrix with values vector return result

        return null;
    }

    /**
     * Checks if a List of Embeddings matches the objects inputEmbeddingSize and inputEmbeddingAmount attribute values
     * @param embeddings A list of the (input) embeddings that should be checked
     * @return true if the Input matches the requirements, false if the input is invalid for this head
     */
    private boolean isValidInput(List<Embedding> embeddings){
        if (embeddings.size() != inputEmbeddingAmount) return false;
        for (Embedding embedding : embeddings){
            if (embedding.getEmbeddingSize() != inputEmbeddingSize) return false;
        }
        return true;
    }

    /**
     * Soft-maxes a vector
     * @param values the vector that will be inserted into to softmax function
     * @return the values soft-maxed
     */
    private static double[] softmax(double[] values) {
        double max = Arrays.stream(values).max().getAsDouble();

        double sum = 0.0;
        double[] result = new double[values.length];

        for (int i = 0; i < values.length; i++) {
            result[i] = Math.exp(values[i] - max);
            sum += result[i];
        }

        for (int i = 0; i < values.length; i++) {
            result[i] /= sum;
        }

        return result;
    }

    /**
     *
     * @param matrix1 
     * @param matrix2
     * @return
     */
    private static double[][] multiplyMatrices(double[][] matrix1, double[][] matrix2){
        if (matrix1[0].length != matrix2.length) return null;

        double[][] resultMatrix = new double[matrix1.length][matrix2[0].length];

        for (int row = 0; row < matrix1.length; row++) {
            for (int col = 0; col < matrix2[0].length; col++) {
                double sum = 0;
                for (int k = 0; k < matrix1[0].length; k++) {
                    sum += matrix1[row][k] * matrix2[k][col];
                }
                resultMatrix[row][col] = sum;
            }
        }

        return resultMatrix;
    }

}
