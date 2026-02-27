package me.damoebe.transformer.embedding;

import java.util.List;

/**
 * A record class for a token, used to construct a transformer internal embedding.
 * @param data The data of this token
 */
public record Token(List<Double> data) {
}
