A. Batches and shards

1. A rank holds a weight gradient with batches={2} and shards={0,1,2,3}. Which call produced this gradient? What does each of the two fields tell you?
2. With 4 ranks, rank 0 holds a full gradient from batch 0 after backward. After scatterreduce, how many batches does the gradient on rank 0 cover, and how many shards does it hold?
3. Activations have batches but no shards. Weights have shards but no batches. Why?

B. The three primitives

4. Each of 2 ranks holds a list of 2 numbers. After allgather, each rank holds 4 numbers. After allreduce, each rank holds 2 numbers. Explain the difference.
5. What goes wrong if you call allreduce on the weight shards?
6. What goes wrong if you call allgather on the one-batch weight gradients?
7. scatterreduce is allreduce followed by a split. Why can a rank not do the split first, to save memory, and then call allreduce?

C. The puzzles as a sequence

8. Gradient accumulation and DDP give the same summed gradient. What does each one cost?
9. In DDP, all ranks hold the same weights after update. Which fact about the allreduce result makes this certain?
10. In Puzzle 3, why does update work on a weight shard, while forward needs the complete weight?
11. In Puzzle 4, your code holds a maximum of one full gradient at a time. What would the maximum be if all ranks ran backward for all layers first, and then called scatterreduce?

D. Memory and scale

12. With 4 ranks, a model has 6 layers and one full weight has size W. For DDP, WSDP, and FSDP, calculate the per-parameter state that one rank holds at the end of the backward pass. Ignore the borrowed weight.
13. You change from 4 ranks to 8 ranks and keep one batch on each rank. In FSDP, which objects on one rank become smaller, and which objects stay the same size?
14. FSDP adds an allgather for each layer in the forward pass and in the backward pass. DDP has no such cost. When is DDP the better selection?

E. Audit

15. A company says that it fine-tuned all the weights of a 120B model with Adam on one machine with 8 GPUs of 80 GB each. Calculate the per-parameter state at 16 bytes per parameter. What must be true for the statement to be possible, and what would you look for in the code?
