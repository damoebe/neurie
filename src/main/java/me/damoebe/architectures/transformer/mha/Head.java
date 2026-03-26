package me.damoebe.architectures.transformer.mha;

import me.damoebe.architectures.Backpropagation;
import me.damoebe.architectures.mlp.structure.Connection;
import me.damoebe.architectures.transformer.embedding.Embedding;
import me.damoebe.architectures.transformer.embedding.Matrix;
import me.damoebe.architectures.transformer.embedding.Sequence;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * The Decoder and Encoder Head Object class
 */
public class Head implements Backpropagation {

    /**
     * All weights of the head in one list, (size = inputEmbeddingSize*inputEmbeddingAmount*3) -> used to
     * calculate the queries, keys and values.
     */
    protected List<double[][]> weights = new ArrayList<>();

    /**
     * All biases of the head in one list, (size = inputEmbeddingSize*inputEmbeddingAmount*3) -> used to
     * calculate the queries, keys and values.
     */
    protected List<double[][]> biases = new ArrayList<>();

    /**
     * The current attention matrix (2d array) -> softmax(dot product/scale + mask)
     */
    private double[][] attention;
    /**
     * The current value matrix. Rows are embeddings, columns are embedding vectors
     */

    private double[][] values;
    private double[][] queries;
    private double[][] keys;

    /**
     * The vector dimension (size) that each inserted embedding must have
     */
    protected final int inputEmbeddingSize;
    /**
     * The amount of embeddings that are allowed to be inserted.
     */
    protected final int inputEmbeddingAmount;

    protected final boolean masked;

    /**
     * The main constructor of the decoder Head
     * Identical to the super constructor
     * @param inputEmbeddingAmount The amount of input embeddings
     * @param inputEmbeddingSize The size of each input embedding
     */
    public Head(int inputEmbeddingAmount, int inputEmbeddingSize, boolean masked) {
        this.inputEmbeddingAmount = inputEmbeddingAmount;
        this.inputEmbeddingSize = inputEmbeddingSize;
        this.masked = masked;

        this.attention = new double[inputEmbeddingAmount][inputEmbeddingAmount];
        this.values = new double[inputEmbeddingAmount][inputEmbeddingSize];

        for (int i = 0; i != 3; i++){
            double[][] weightMatrix = new double[inputEmbeddingSize][inputEmbeddingAmount];
            double[][] biasMatrix = new double[inputEmbeddingSize][inputEmbeddingAmount];
            for (int row = 0; row != inputEmbeddingSize; row++){
                for (int colum = 0; colum != inputEmbeddingAmount; colum++){
                    weightMatrix[row][colum] = Connection.getRandomWeight();
                    biasMatrix[row][colum] = Math.random() * 2 - 1;
                }
            }
            weights.add(weightMatrix);
            biases.add(biasMatrix);
        }
    }

    /**
     * Inserts an input sequence into this head and calculated the attention
     * @param inputEmbeddingLists The input Embedding lists -> here it can only contain one embedding-list
     * @throws Exception if the inserted input embeddings don't fit the expectations
     */
    public void insertInput(Sequence... inputEmbeddingLists) throws Exception{
        if (inputEmbeddingLists.length != 1) throw new Exception("This type of head can only take one Embedding list.");

        if (!isValidInput(inputEmbeddingLists[0])) throw new Exception("Not a valid input Embedding list!");

        List<double[][]> QKVMatrices = generateQKVMatrices(inputEmbeddingLists);

        final double[][] queries = QKVMatrices.getFirst();
        final double[][] keys = QKVMatrices.get(1);

        this.values = QKVMatrices.getLast();
        this.keys = keys;
        this.queries = queries;

        double[][] scaled_masked_dot_product = new double[queries.length][keys.length];
        for (int q = 0; q != queries.length; q++){
            for (int k = 0; k != keys.length; k++){
                // triangular mask (causal mask) only if masked is true
                if (masked) {
                    if (k > q) {
                        scaled_masked_dot_product[q][k] = Double.NEGATIVE_INFINITY;
                        continue;
                    }
                }

                // calculate dot product (sum(vec*vec))
                double sum = 0;
                double[] query = queries[q];
                double[] key = keys[k];
                for (int i = 0; i != query.length; i++){
                    sum += query[i] * key[i];
                }
                scaled_masked_dot_product[q][k] = sum / Math.sqrt(key.length); // scaling
            }
        }

        double[][] attention = new double[queries.length][keys.length];
        for (int row = 0; row != queries.length; row++){
            attention[row] = softmax(scaled_masked_dot_product[row]);
        }
        this.attention = attention;
    }

    /**
     * Generates a list of query, key and value matrices based on the heads weights and biases.
     * @param inputEmbeddingLists The input Embedding lists -> here it can only contain one embedding-list
     * @return a list of query, key and value matrices
     */
    protected List<double[][]> generateQKVMatrices(Sequence[] inputEmbeddingLists) throws Exception{
        List<double[][]> QKV = new ArrayList<>();

        // bad performance fix in future
        double[][] queries = Matrix.add(
                Objects.requireNonNull(Matrix.multiply(inputEmbeddingLists[0].getVerticalMatrix(), weights.getFirst())),
                biases.getFirst()
        );

        double[][] keys = Matrix.add(
                Objects.requireNonNull(Matrix.multiply(inputEmbeddingLists[0].getVerticalMatrix(), weights.get(1))),
                biases.get(1)
        );

        double[][] values = Matrix.add(
                Objects.requireNonNull(Matrix.multiply(inputEmbeddingLists[0].getVerticalMatrix(), weights.getLast())),
                biases.getLast()
        );

        assert queries != null && keys != null && values != null;

        double[][] verticalQueries = Matrix.transpose(queries);
        double[][] verticalKeys = Matrix.transpose(keys);
        double[][] verticalValues = Matrix.transpose(values);

        QKV.add(verticalQueries);
        QKV.add(verticalKeys);
        QKV.add(verticalValues);

        return QKV;
    }

    /**
     * Updates all QKV weights based on delta matrix from MultiHeadAttention weights
     * @param deltas The delta from the mha of this head
     * @return The new deltas for the next backpropagation step
     */
    public double[][] backPropagate(double[][] deltas){
        // TODO: update deltas and weights based on head inserted deltas using the chain-rule and deriving softmax & co
        // partial derive head function with respect to weights (use multiply matrices and transpose)
        return null;
    }

    /**
     * Generates an output of this head by using the current attention, which is multiplied with the current value-vectors
     * @return The output for the input Embeddings as a matrix.
     */
    public double[][] getOutput(){
        return Matrix.multiply(values, attention);
    }

    /**
     * Checks if a List of Embeddings matches the objects inputEmbeddingSize and inputEmbeddingAmount attribute values
     * @param embeddings A list of the (input) embeddings that should be checked
     * @return true if the Input matches the requirements, false if the input is invalid for this head
     */
    private boolean isValidInput(Sequence embeddings){
        if (embeddings.embeddings().size() != inputEmbeddingAmount) return false;
        for (Embedding embedding : embeddings.embeddings()){
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

    public int getInputEmbeddingSize() {
        return inputEmbeddingSize;
    }

    public int getInputEmbeddingAmount() {
        return inputEmbeddingAmount;
    }
}
