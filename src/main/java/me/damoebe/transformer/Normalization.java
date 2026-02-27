package me.damoebe.transformer;

import me.damoebe.mlp.structure.Connection;
import me.damoebe.transformer.embedding.Embedding;
import me.damoebe.transformer.embedding.Sequence;

import java.util.ArrayList;

import java.util.List;

/**
 * A class for Sub-Layer normalisation, it stores weights and biases.
 */
public class Normalization {

    private final List<Double> normalizationWeights = new ArrayList<>();
    private final List<Double> normalisationBiases = new ArrayList<>();

    /**
     * Main constructor for Normalisation class
     * @param embeddingAmount The embedding amount
     * @param embeddingSize The embedding size (*2 if two sequences)
     */
    public Normalization(int embeddingAmount, int embeddingSize){
        for (int embedding = 0; embedding != embeddingAmount; embedding++){
            for (int value = 0; value != embeddingSize; value++){
                normalizationWeights.add(Connection.getRandomWeight());
                normalisationBiases.add(Connection.getRandomBias());
            }
        }
    }
    /**
     * Normalize 1 or 2 embeddings
     * @param embeddings The embedding-lists that should be normalized.
     * @return The normalized embedding-lists
     */
    public final Sequence[] normalize(Sequence... embeddings){
        Sequence[] resultEmbeddings = new Sequence[embeddings.length];
        int sequenceIndex = 0;
        for (Sequence sequence : embeddings){
            Sequence normalizedSequence = new Sequence();
            for (Embedding embedding : sequence.embeddings()){
                List<Double> normalizedData = getNormalizedEmbedding(embedding);
                normalizedSequence.embeddings().add(new Embedding(normalizedData));
            }
            resultEmbeddings[sequenceIndex] = normalizedSequence;
            sequenceIndex++;
        }
        return resultEmbeddings;
    }

    /**
     * Normalized the vector of a single embedding.
     * @param embedding The embedding that should be normalized
     * @return The normalized embedding vector as a Double list
     */
    private List<Double> getNormalizedEmbedding(Embedding embedding) {
        double avg = 0;
        for (Double value : embedding.data()){
            avg += value;
        }
        avg /= embedding.data().size();
        double o_sq = 0;
        for (Double value : embedding.data()){
            o_sq += (value - avg) * (value -avg);
        }
        List<Double> normalizedData = new ArrayList<>();
        int i = 0;
        for (Double value : embedding.data()){
            normalizedData.add(normalizationWeights.get(i) * ((value-avg)/Math.sqrt(o_sq)) + normalisationBiases.get(i));
            i++;
        }
        return normalizedData;
    }
}
