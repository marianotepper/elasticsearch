/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the "Elastic License
 * 2.0", the "GNU Affero General Public License v3.0 only", and the "Server Side
 * Public License v 1"; you may not use this file except in compliance with, at
 * your election, the "Elastic License 2.0", the "GNU Affero General Public
 * License v3.0 only", or the "Server Side Public License, v 1".
 */

package org.elasticsearch.search.query;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.CollectionTerminatedException;
import org.apache.lucene.search.FilterLeafCollector;
import org.apache.lucene.search.LeafCollector;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.TotalHitCountCollector;
import org.apache.lucene.util.ThreadInterruptedException;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Extension of {@link TotalHitCountCollector} that supports early termination of total hits counting based on a provided threshold.
 * Note that the total hit count may be retrieved from {@link org.apache.lucene.search.Weight#count(LeafReaderContext)},
 * in which case early termination is only applied to the leaves that do collect documents.
 */
class PartialHitCountCollector extends TotalHitCountCollector {

    private final HitsThresholdChecker hitsThresholdChecker;
    // Non-null only when the search may target more than one partition of the same segment, i.e. when
    // ContextIndexSearcher#computeSlices split an oversized segment into several LeafReaderContextPartitions
    // (see #153083). Coordinates across the collector instances (one per slice, potentially running
    // concurrently) so that a segment split into partitions only ever contributes its count once: without
    // this, Weight#count(LeafReaderContext) -- which returns the count for the *whole* segment, oblivious to
    // partition bounds -- would otherwise be added once per partition. Mirrors the coordination Lucene's own
    // TotalHitCountCollectorManager applies for the same reason.
    private final Map<Object, Future<Boolean>> earlyTerminatedMap;
    private boolean earlyTerminated;

    PartialHitCountCollector(HitsThresholdChecker hitsThresholdChecker) {
        this(hitsThresholdChecker, null);
    }

    PartialHitCountCollector(HitsThresholdChecker hitsThresholdChecker, Map<Object, Future<Boolean>> earlyTerminatedMap) {
        this.hitsThresholdChecker = hitsThresholdChecker;
        this.earlyTerminatedMap = earlyTerminatedMap;
    }

    @Override
    public ScoreMode scoreMode() {
        return hitsThresholdChecker.totalHitsThreshold == Integer.MAX_VALUE ? super.scoreMode() : ScoreMode.TOP_DOCS;
    }

    @Override
    public LeafCollector getLeafCollector(LeafReaderContext context) throws IOException {
        if (hitsThresholdChecker.totalHitsThreshold == Integer.MAX_VALUE) {
            return countingLeafCollector(context);
        }
        earlyTerminateIfNeeded();
        return new FilterLeafCollector(countingLeafCollector(context)) {
            @Override
            public void collect(int doc) throws IOException {
                earlyTerminateIfNeeded();
                hitsThresholdChecker.incrementHitCount();
                super.collect(doc);
            }
        };
    }

    /**
     * Delegates to {@link TotalHitCountCollector#getLeafCollector(LeafReaderContext)}, which may throw
     * {@link CollectionTerminatedException} right away if {@code Weight#count} can answer the whole segment's
     * count without collecting. When {@link #earlyTerminatedMap} is set, only the first partition of a given
     * leaf actually makes that call; later partitions of the same leaf replay its outcome (both the count and
     * the early termination) instead of repeating -- and so redundantly counting -- the whole-segment shortcut.
     */
    private LeafCollector countingLeafCollector(LeafReaderContext context) throws IOException {
        if (earlyTerminatedMap == null) {
            return super.getLeafCollector(context);
        }
        Future<Boolean> earlyTerminated = earlyTerminatedMap.get(context.id());
        if (earlyTerminated == null) {
            CompletableFuture<Boolean> firstEarlyTerminated = new CompletableFuture<>();
            Future<Boolean> previousEarlyTerminated = earlyTerminatedMap.putIfAbsent(context.id(), firstEarlyTerminated);
            if (previousEarlyTerminated == null) {
                // first partition of this leaf gets to decide what subsequent partitions of it do
                try {
                    LeafCollector leafCollector = super.getLeafCollector(context);
                    firstEarlyTerminated.complete(false);
                    return leafCollector;
                } catch (CollectionTerminatedException e) {
                    firstEarlyTerminated.complete(true);
                    throw e;
                }
            }
            earlyTerminated = previousEarlyTerminated;
        }
        try {
            if (earlyTerminated.get()) {
                // the first partition of this leaf got its count from Weight#count and terminated right away;
                // do the same here rather than counting (part of) the same segment a second time
                throw new CollectionTerminatedException();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ThreadInterruptedException(e);
        } catch (ExecutionException e) {
            throw new RuntimeException(e);
        }
        // the first partition of this leaf couldn't shortcut and is collecting docs for real; do the same here
        return createLeafCollector();
    }

    private void earlyTerminateIfNeeded() {
        if (hitsThresholdChecker.isThresholdReached()) {
            earlyTerminated = true;
            throw new CollectionTerminatedException();
        }
    }

    boolean hasEarlyTerminated() {
        return earlyTerminated;
    }

    static class HitsThresholdChecker {
        private final int totalHitsThreshold;
        private final AtomicInteger numCollected = new AtomicInteger();

        HitsThresholdChecker(int totalHitsThreshold) {
            this.totalHitsThreshold = totalHitsThreshold;
        }

        void incrementHitCount() {
            numCollected.incrementAndGet();
        }

        boolean isThresholdReached() {
            return numCollected.getAcquire() >= totalHitsThreshold;
        }
    }
}
