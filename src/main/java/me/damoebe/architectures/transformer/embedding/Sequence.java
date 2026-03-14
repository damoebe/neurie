package me.damoebe.architectures.transformer.embedding;

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
     * Constructs a Sequence from an embedding list (used for sub-sequence splitting)
     * @param embeddings The List of Embeddings from which this Sequence should be constructed
     */
    public Sequence(List<Embedding> embeddings){
        this.embeddings = embeddings;
    }

    /**
     * Generate a 'raw' Double list for this sequence containing all embedding values
     * @return The raw Double list
     */
    public List<Double> getRawList(){
        List<Double> rawList = new ArrayList<>();
        for (Embedding embedding : this.embeddings){
            rawList.addAll(embedding.data());
        }
        return rawList;
    }

    /**
     * Gets the horizontal matrix of this sequence (rows -> embeddingData, columns -> embedding)
     * @return The horizontal Matrix
     */
    public double[][] getHorizontalMatrix(){
        return Matrix.transpose(getVerticalMatrix());
    }

    /**
     * Gets the vertical (standard dimensions) matrix of this sequence where rows -> embeddings and
     * columns -> embeddingData
     * @return The vertical Matrix
     */
    public double[][] getVerticalMatrix(){
        double[][] verticalMatrix = new double[embeddings.size()][embeddings.getFirst().getEmbeddingSize()];
        for (int embeddingIndex = 0; embeddingIndex != embeddings.size(); embeddingIndex++){
            for (int dataIndex = 0; dataIndex != embeddings.getFirst().getEmbeddingSize(); dataIndex++){
                verticalMatrix[embeddingIndex][dataIndex] = embeddings.get(embeddingIndex).data().get(dataIndex);
            }
        }
        return verticalMatrix;
    }

    /**
     * Splits this sequences into n sub-sequences
     * @param subSequenceAmount The amount of sub-sequences this sequence should be split into
     * @return A list of all sub-sequences of this sequence
     */
    public List<Sequence> getSubSequences(int subSequenceAmount){
        List<Sequence> subSequences = new ArrayList<>();
        for (int subSequence = 0; subSequence != subSequenceAmount; subSequence++){
            List<Embedding> subSequenceEmbeddings = new ArrayList<>();
            for (int embedding = 0; embedding != embeddings.size()/subSequenceAmount; embedding++){
                subSequenceEmbeddings.add(new Embedding(embeddings.get(embedding).data()));
            }
            subSequences.add(new Sequence(subSequenceEmbeddings));
        }
        return subSequences;
    }

    /**
     * Getter for the stored embeddings
     * @return A list with all embeddings
     */
    public List<Embedding> embeddings(){
        return embeddings;
    }
}
