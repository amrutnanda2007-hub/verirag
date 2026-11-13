# Enterprise Distributed Systems Architecture Guide

## 1. Consensus Protocols & Fault Tolerance

Distributed systems rely on consensus algorithms to maintain a consistent replicated state machine across multiple nodes in the presence of network partitions and node failures.

### The Raft Consensus Algorithm
Raft achieves consensus by electing a distinguished leader node that accepts client requests, appends log entries to its local write-ahead log (WAL), and replicates these entries across a majority quorum of followers. 

- **Leader Election:** Raft uses randomized election timers (typically between 150ms and 300ms) to avoid split votes. When a follower node does not receive heartbeats (empty `AppendEntries` RPCs) before its election timer expires, it transitions to the candidate state, increments its term number, votes for itself, and broadcasts `RequestVote` RPCs to all peers.
- **Log Replication:** Once a leader receives a consensus quorum vote (i.e., $\lfloor N/2 \rfloor + 1$ affirmative responses), log entries are marked as committed. The state machine applies committed entries deterministically in log order.
- **Handling Network Partitions:** In the event of a network partition creating an asymmetric split (e.g., 2 nodes on side A, 3 nodes on side B in a 5-node cluster), only the majority partition (side B) can commit new log entries. The minority partition continues to accept read-only queries with linearizable lease checks or rejects writes until partition resolution.

## 2. Distributed Caching & Cache Invalidation

Maintaining consistency between primary transactional databases and in-memory caching tiers (such as Redis or Memcached) is notoriously difficult.

### The Cache-Aside (Lazy Loading) Pattern
In the cache-aside pattern, application code first queries the cache. If a cache miss occurs, the application reads the record directly from the persistent datastore, writes the result into the cache with an explicit Time-To-Live (TTL), and returns the payload to the caller.

### Write-Through vs Write-Behind
- **Write-Through:** Synchronously writes updates to both the cache and underlying database within the same transaction boundary. Guarantees consistency at the cost of higher write latency.
- **Write-Behind (Write-Back):** Writes immediately to the caching tier and queues asynchronous batched writes to the database. Delivers microsecond write latencies but introduces data loss risk if the cache node crashes before flushing dirty pages.

## 3. High-Throughput Event Streaming Architecture

Event-driven architectures decouple producers from consumers using partitioned commit logs.

### Partitioning & Consumer Groups
Apache Kafka and Pulsar distribute events across partitioned topics based on hashing keys. Consumer groups enable horizontal scaling by assigning individual partitions to single consumer instances within a group. If a consumer crashes, a group rebalance reassigns its assigned partitions to surviving peers without halting global topic ingestion.
