import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell(column=0)
def _():
    import asyncio

    import chalk
    import marimo as mo

    from drawing import draw, draw_group
    from lib import Dist, Model, WeightGrad

    chalk.set_svg_height(400)
    chalk.set_svg_draw_height(600)
    return Dist, Model, WeightGrad, asyncio, chalk, draw, draw_group, mo


@app.cell
def _(mo):
    mo.md(r"""
    # LLM Training Puzzles

    by Sasha Rush ([@srush_nlp](https://twitter.com/srush_nlp))
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Preliminaries

    The goal of these puzzles is to learn about distributed training of LLMs. However, we will be primarily concerned with a speed and memory efficiency of completing a single update of the models. To make things simpler, we will abstract away from the standard tensor-based transformer model, and just consider a state-less representation of each of the components of a multi-layer neural network.
    """)
    return


@app.cell
def _(Model):
    model = Model(layers=2, batches=4)
    weights, opt_states, activations, grad_activations, grad_weights = model.storage()
    return (
        activations,
        grad_activations,
        grad_weights,
        model,
        opt_states,
        weights,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Our library has 5 parts:

    * Weights
    * Optimizer States - Values needed to update the weights
    * Activations - The internal values computed on the forward pass
    * Grad Activations - The gradients of the loss wrt to activations, needed for backward pass
    * Grad Weights - The gradients of the loss wrt to weights, needed for updates

    For these puzzles, you are *not allowed* to have local variables. You need to store each of these in the dictionary corresponding to its type.

    We begin by tracing the lifecycle of a single model update.
    """)
    return


@app.cell
def _(activations, model):
    # Get the input activations to the model for batches 2, 3
    activations[0] = model.get_activation(batches=[2, 3])
    activations[0]
    return


@app.cell
def _(model, opt_states, weights):
    # Load the weights (random) for layers 0 and 1
    for _i in range(model.LAYERS):
        weights[_i], opt_states[_i] = model.load_weights(_i)
    weights[0]
    return


@app.cell
def _(activations, model, weights):
    # Activations can be moved forward a layer if you have the weights.
    activations[1] = model.forward(layer=0, inp=activations[0], weight=weights[0])
    activations[2] = model.forward(layer=1, inp=activations[1], weight=weights[1])
    activations[1]
    return


@app.cell
def _(activations, draw_group):
    # Draw all the current activations in memory.
    draw_group(activations)
    return


@app.cell
def _(activations, grad_activations, model):
    # At the last layer, we can convert an activation to a grad activation by calling `loss`
    grad_activations[model.LAYERS] = model.loss(activations[model.LAYERS])
    grad_activations[model.LAYERS]
    return


@app.cell
def _(activations, grad_activations, grad_weights, model, weights):
    # Calling `backward` requires the forward activation, the backward grad activation, and the weights.
    # It returns the grad weights and the backward activation.
    grad_weights[1], grad_activations[1] = model.backward(1, activations[1], grad_activations[2], weights[1])
    grad_weights[0], grad_activations[0] = model.backward(0, activations[0], grad_activations[1], weights[0])
    grad_activations[1]
    return


@app.cell
def _(draw_group, grad_activations, model):
    # We can use delete to remove any memory that is not longer needed.
    print("Before memory:", model.memory())
    del grad_activations[1]
    print("After memory:", model.memory())
    model.status()
    draw_group(grad_activations)
    return


@app.cell
def _(draw_group, grad_weights):
    # Grad weights keep track of which batches they are for. Here we only have the grad weights for batches 2 and 3.
    draw_group(grad_weights)
    return


@app.cell
def _(grad_weights, model, opt_states, weights):
    # If we try to update with the grad weights we will get an error.
    try:
        model.update(0, weight_grad=grad_weights[0], weight=weights[0], opt_state=opt_states[0])
    except AssertionError as e:
        print("Error! Only have batches")
        print(e)
    return


@app.cell
def _(grad_weights, model):
    # For this example, we can cheat. Pretend we had the other gradients we needed.
    grad_weights[0, 0] = model.fake_grad(0, [0, 1])
    grad_weights[1, 0] = model.fake_grad(1, [0, 1])
    grad_weights[0, 0]
    return


@app.cell
def _(grad_weights):
    # Summing together grad_weights gives the full gradient.
    grad_weights[0] = grad_weights[0] + grad_weights[0, 0]
    return


@app.cell
def _(grad_weights, model, opt_states, weights):
    # Now we can call update to the get the new weights and opt_state.
    weights[0], opt_states[0] = model.update(
        0, weight_grad=grad_weights[0], weight=weights[0], opt_state=opt_states[0]
    )

    # WARNING: You need to set all variables. Otherwise they are not counted towards memory.
    grad_weights[1] = grad_weights[1] + grad_weights[1, 0]
    weights[1], opt_states[1] = model.update(
        1, weight_grad=grad_weights[1], weight=weights[1], opt_state=opt_states[1]
    )
    return


@app.cell
def _(Model, draw_group, model, weights):
    # We can complete the tests by setting these as the final weights and calling check.
    model.set_final_weight(0, weights[0])
    model.set_final_weight(1, weights[1])
    Model.check([model])
    draw_group(model.final_weights)
    return


@app.cell
def _(draw, model):
    # We can view the final outcome of the system as a diagram.
    # This show the forward and backward passes (numbers of batches) and the updates.
    # The lines on the bottom show the memory that is used at each time step.
    draw([model])
    return


@app.cell(column=1)
def _(mo):
    mo.md(r"""
    ### Puzzle 0 - Standard Training

    Write a standard (non-distributed) training loop that acts on all the batches and loads all the weights. It should just run forward, loss, backward, and update. Aim for the least amount of max memory used.

    * Target Time:  17 steps
    * Target Memory: 2600000
    """)
    return


@app.cell
def _(Model):
    def basic(model: Model) -> Model:
        # Storage on device.
        weights, opt_states, activations, grad_activations, grad_weights = model.storage()

        # Load in the full weights
        for l in range(model.LAYERS):
            weights[l], opt_states[l] = model.load_weights(l)

        # Load the input layer activations
        activations[0] = model.get_activation(range(model.BATCHES))

        for l_ind in range(model.LAYERS):
            activations[l_ind + 1] = model.forward(layer=l_ind, inp=activations[0], weight=weights[0])
            activations[2] = model.forward(layer=1, inp=activations[1], weight=weights[1])

        assert False, 'TODO: Implement me'

        for l in range(model.LAYERS):
            model.set_final_weight(l, weights[l])
        return model

    return (basic,)


@app.cell
def _(Dist, Model, basic, draw_group):
    basic_out = basic(Model(layers=2, batches=4, rank=0, dist=Dist(1)))
    draw_group(basic_out.final_weights)
    return (basic_out,)


@app.cell
def _(basic_out, draw):
    draw([basic_out])
    return


@app.cell
def _(Model, basic_out):
    Model.check([basic_out])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Puzzle 1 - Gradient Accumulation

    For this puzzle, the goal is to reduce max memory usage. To do so you are going to run on each batch individually instead of all together.

    Write a function with four parts. First run on batches {0} and then {1} etc. Sum the grad weights and then update.

    * Target Time:  17 steps
    * Target Memory: 2000000
    """)
    return


@app.cell
def _(Model):
    def grad_accum(model: Model) -> Model:
        # Storage on device.
        weights, opt_states, activations, grad_activations, grad_weights = model.storage()

        # Load in the full weights
        for l in range(model.LAYERS):
            weights[l], opt_states[l] = model.load_weights(l)

        assert False, 'TODO: Implement me'
        for l in range(model.LAYERS):
            model.set_final_weight(l, weights[l])
        return model

    return (grad_accum,)


@app.cell
def _(Dist, Model, draw_group, grad_accum):
    grad_accum_out = grad_accum(Model(layers=2, batches=4, rank=0, dist=Dist(1)))
    draw_group(grad_accum_out.final_weights)
    return (grad_accum_out,)


@app.cell
def _(draw, grad_accum_out):
    draw([grad_accum_out])
    return


@app.cell
def _(Model, grad_accum_out):
    Model.check([grad_accum_out])
    return


@app.cell(column=2)
def _(mo):
    mo.md(r"""
    ## Communications: AllReduce

    When working with multiple GPUs we need to have communication.
    The primary communication primitives for GPUs are implemented in NCCL.

    https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/operations.html

    We are not going to use these directly, but simulate them using Python and asyncio.

    The first operation is AllReduce. We will have 4 GPUs (ranks=4) and use them each to compute a batch of weight grads.
    """)
    return


@app.cell
def _(WeightGrad):
    ranks = 4
    weight_grads = [WeightGrad(0, 1, {i}, ranks) for i in range(ranks)]
    weight_grads[0] + weight_grads[1] + weight_grads[2] + weight_grads[3]
    return ranks, weight_grads


@app.cell
def _(Model, WeightGrad, weight_grads):
    # Simple asynchronous function that calls allreduce to sum the weight grads at layer 0
    async def myfunc(model: Model) -> WeightGrad:
        return await model.allreduce(weight_grads[model.rank], 0)

    return (myfunc,)


@app.cell
async def _(Dist, Model, asyncio, myfunc, ranks):
    # This code uses asyncio to run the above function on 4 "GPUs" .
    _dist = Dist(ranks)
    out_weight_grads = await asyncio.gather(*[
        myfunc(Model(layers=1, batches=1, rank=i, dist=_dist))
        for i in range(ranks)])
    out_weight_grads[0]
    return


@app.cell
def _(mo):
    mo.md(r"""
    Note: When running communication operations like AllReduce on a GPU, the communication happens in parallel to the computation on that GPU. That means the API for AllReduce does not block, and allows the model to continue running while waiting for this command to run. This means it is beneficial to run AllReduce (and other communication) as early as possible so that other compute can be run during the reduction.

    We will ignore this in these puzzles and represent communication as happening efficiently.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Puzzle 2 - Distributed Data Parallel

    Write a function with four parts. First run on batches {0} and then {1} etc. Sum the grad weights and then update. The main benefit of this approach is compute efficiency over gradient accumulation.

    * Total Steps: 5
    * Total Memory: 1800000
    """)
    return


@app.cell
def _(Model):
    async def ddp(model: Model) -> Model:
        # Storage on device.
        weights, opt_states, activations, grad_activations, grad_weights = model.storage()
        # Load all the activations
        model.activations[0] = model.get_activation([model.rank])

        assert False, 'TODO: Implement me'
        for l in range(model.LAYERS):
            model.set_final_weight(l, weights[l])
        return model

    return (ddp,)


@app.cell
async def _(Dist, Model, asyncio, ddp, draw_group, ranks):
    _dist = Dist(ranks)
    ddp_out = await asyncio.gather(*[
        ddp(Model(layers=2, batches=ranks, rank=i, dist=_dist))
        for i in range(ranks)])
    draw_group(ddp_out[0].final_weights)
    return (ddp_out,)


@app.cell
def _(ddp_out, draw):
    draw(ddp_out)
    return


@app.cell
def _(Model, ddp_out):
    Model.check(ddp_out)
    return


@app.cell(column=3)
def _(mo):
    mo.md(r"""
    ## Communication: AllGather / Sharding

    Our next primitive is AllGather. This allows us to communicate "shards" of an object stored on different GPUs to all the GPUs.
    """)
    return


@app.cell
def _(Dist, Model):
    # Load only part of a weights.
    shard_model = Model(layers=2, batches=1, rank=0, dist=Dist(1))
    weight_shard, _ = shard_model.load_weights(0, shard=0, total=4)
    weight_shard
    return (shard_model,)


@app.cell
def _(ranks, shard_model):
    # Combine togegher two shards on one machine.
    weight_shards = [shard_model.load_weights(0, shard=i, total=ranks)[0] for i in range(ranks)]
    weight_shards[0].combine(weight_shards[2])
    return (weight_shards,)


@app.cell
async def _(Dist, Model, WeightGrad, asyncio, ranks, weight_shards):
    # Use allgather to collect the shards from all machines.
    async def mygather(model: Model) -> WeightGrad:
        # Allreduce sums together all the weight grads
        return await model.allgather(weight_shards[model.rank], 0)

    _dist = Dist(ranks)
    out_weights = await asyncio.gather(*[
        mygather(Model(layers=1, batches=1, rank=i, dist=_dist))
        for i in range(ranks)])
    out_weights[0]
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Puzzle 3: Weight-Sharded Data Parallel

    Run a model that shards each layer weight over all the machines. Reconstruct the layer weight at each layer using allgather. Finally update the weights on each machine using allreduce.

    * Total Steps: 20
    * Total Memory: 2800000
    """)
    return


@app.cell
def _(Model):
    async def wsdp(model: Model) -> Model:
        # Storage on device.
        weights, opt_states, activations, grad_activations, grad_weights = model.storage()

        # Load all the activations
        model.activations[0] = model.get_activation([model.rank])

        # Load a shard of the weights for every layer. Load in the full weights
        for l in range(model.LAYERS):
            weights[l], opt_states[l] = model.load_weights(l, model.rank, model.RANKS)

        assert False, 'TODO: Implement me'
        for l in range(model.LAYERS):
            model.set_final_weight(l, weights[l])

        return model

    return (wsdp,)


@app.cell
async def _(Dist, Model, asyncio, draw_group, ranks, wsdp):
    _dist = Dist(ranks)
    wsdp_out = await asyncio.gather(*[
        wsdp(Model(layers=6, batches=ranks, rank=i, dist=_dist))
        for i in range(ranks)])
    draw_group(wsdp_out[1].final_weights)
    return (wsdp_out,)


@app.cell
def _(draw, wsdp_out):
    draw(wsdp_out)
    return


@app.cell
def _(Model, wsdp_out):
    Model.check(wsdp_out)
    return


@app.cell(column=4)
def _(mo):
    mo.md(r"""
    ## Communication: Scatter-Reduce

    Scatter across shards
    Reduce across batches
    """)
    return


@app.cell
def _(WeightGrad):
    grad_weight = WeightGrad(0, 1, batches={1}, total_batches=4,
                             shards={1}, total=4)
    grad_weight
    return


@app.cell
def _(WeightGrad):
    unscattered_grads = {i: WeightGrad(0, 1, batches={i}, total_batches=4,
                                       shards={0, 1, 2, 3}, total=4) for i in range(4)}
    unscattered_grads[2]
    return (unscattered_grads,)


@app.cell
async def _(Dist, Model, WeightGrad, asyncio, ranks, unscattered_grads):
    async def scatterreduce(model: Model) -> WeightGrad:
        # Allreduce sums together all the weight grads
        return await model.scatterreduce(unscattered_grads[model.rank], 0)

    _dist = Dist(ranks)
    scattered_grads = await asyncio.gather(*[
        scatterreduce(Model(layers=1, batches=1, rank=i, dist=_dist))
        for i in range(ranks)])
    scattered_grads[0]
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Puzzle 4: Fully-Sharded Data Parallel

    Run a model that shards each layer weight over all the machines. Reconstruct the layer weight at each layer using allgather. Collect the gradients with scatter-reduce.

    * Total Steps: 20
    * Total Memory: 2300000
    """)
    return


@app.cell
def _(Model):
    async def fsdp(model: Model) -> Model:
        # Storage on device.
        weights, opt_states, activations, grad_activations, grad_weights = model.storage()

        # Load all the activations
        model.activations[0] = model.get_activation([model.rank])

        # Load a shard of the weights for every layer. Load in the full weights
        for l in range(model.LAYERS):
            weights[l], opt_states[l] = model.load_weights(l, model.rank, model.RANKS)

        assert False, 'TODO: Implement me'
        for l in range(model.LAYERS):
            model.set_final_weight(l, weights[l])
        return model

    return (fsdp,)


@app.cell
async def _(Dist, Model, asyncio, draw_group, fsdp, ranks):
    _dist = Dist(ranks)
    fsdp_out = await asyncio.gather(*[
        fsdp(Model(layers=6, batches=ranks, rank=i, dist=_dist))
        for i in range(ranks)])
    draw_group(fsdp_out[1].final_weights)
    return (fsdp_out,)


@app.cell
def _(draw, fsdp_out):
    draw(fsdp_out)
    return


@app.cell
def _(Model, fsdp_out):
    Model.check(fsdp_out)
    return


@app.cell(column=5)
def _(mo):
    mo.md(r"""
    ## Communication: Point-to-Point

    An alternative approach to communication is to directly communicate specific information between GPUs. In our model, both GPUs talking to each other block and wait for the handoff.
    """)
    return


@app.cell
async def _(Dist, Model, asyncio):
    async def talk(model: Model) -> None:
        if model.rank == 0:
            await model.pass_to(1, "extra cheese")
            val = await model.receive()
            print(val)
        else:
            val = await model.receive()
            print(val)
            val = await model.pass_to(0, "pizza")

    _dist = Dist(2)
    await asyncio.gather(*[
        talk(Model(layers=1, batches=1, rank=i, dist=_dist))
        for i in range(2)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Puzzle 5: Pipeline Parallelism

    Split the layer weights and optimizers equally between GPUs. Have each GPU handle only its layer. Pass the full set of batches for activations and grad_activations between layers using p2p communication. No need for any global communication.

    * Total Steps: 66
    * Total Memory: 3300000
    """)
    return


@app.cell
def _(Model):
    async def pipeline(model: Model) -> Model:
        weights, opt_states, activations, grad_activations, grad_weights = model.storage()
        per_rank = model.LAYERS // model.RANKS
        my_layers = list([l + (model.rank * per_rank) for l in range(per_rank)])
        for l in my_layers:
            weights[l], opt_states[l] = model.load_weights(l)
        assert False, 'TODO: Implement me'
        for l in my_layers:
            model.set_final_weight(l, weights[l])
        return model

    return (pipeline,)


@app.cell
async def _(Dist, Model, asyncio, draw_group, pipeline, ranks):
    _dist = Dist(ranks)
    pipeline_out = await asyncio.gather(*[
        pipeline(Model(layers=8, batches=ranks, rank=i, dist=_dist))
        for i in range(ranks)])
    draw_group(pipeline_out[1].final_weights)
    return (pipeline_out,)


@app.cell
def _(draw, pipeline_out):
    draw(pipeline_out)
    return


@app.cell
def _(Model, pipeline_out):
    Model.check(pipeline_out)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Puzzle 6: GPipe Schedule

    A major issue with the pipeline approach is that it causes a "bubble", i.e. time in the later layers waiting for the earlier layers to complete. An alternative approach is to split the batches smaller so you can pass them earlier.

    In this puzzle, you should run each batch by itself, and then pass. The graph should look similar as the one above but with a smaller bubble.

    * Total Steps: 33
    * Total Memory: 4100000
    """)
    return


@app.cell
def _(Model):
    async def gpipe(model: Model) -> Model:
        weights, opt_states, activations, grad_activations, grad_weights = model.storage()
        per_rank = model.LAYERS // model.RANKS
        my_layers = list([l + (model.rank * per_rank) for l in range(per_rank)])
        for l in my_layers:
            weights[l], opt_states[l] = model.load_weights(l)

        assert False, 'TODO: Implement me'
        for l in my_layers:
            model.set_final_weight(l, weights[l])

        return model

    return (gpipe,)


@app.cell
async def _(Dist, Model, asyncio, draw_group, gpipe, ranks):
    _dist = Dist(ranks)
    gpipe_out = await asyncio.gather(*[
        gpipe(Model(layers=8, batches=ranks, rank=i, dist=_dist))
        for i in range(ranks)])
    draw_group(gpipe_out[1].final_weights)
    return (gpipe_out,)


@app.cell
def _(draw, gpipe_out):
    draw(gpipe_out)
    return


@app.cell
def _(Model, gpipe_out):
    Model.check(gpipe_out)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Puzzle 7: Pipeline + FSDP

    As a last exercise, we can put everything together. Here we are going to run a combination of pipeline parallelism while also sharding our weight between 16 different machines. Here the model only has 4 layers, so we will assign 4 GPUs to each layer in the pipeline parallel approach.

    This example requires combining both collective communication and p2p communication effectively.

    * Total Steps: 15
    * Total Memory: 1000000
    """)
    return


@app.cell
def _(Model):
    async def pipeline_fsdp(model: Model) -> Model:
        weights, opt_states, activations, grad_activations, grad_weights = model.storage()
        per_rank = model.LAYERS // (model.RANKS // 4)
        my_layers = list([l + ((model.rank % 4)  * per_rank) for l in range(per_rank)])
        for l in range(model.LAYERS):
            weights[l, 0], opt_states[l, 0] = model.load_weights(l, model.rank, model.RANKS)
        def empty_grad(l):
            return model.fake_grad(l, [])
        assert False, 'TODO: Implement me'
        for l in range(model.LAYERS):
            model.set_final_weight(l, weights[l])
        # Update
        return model

    return (pipeline_fsdp,)


@app.cell
async def _(Dist, Model, asyncio, pipeline_fsdp, ranks):
    _dist = Dist(16)
    pipeline_fsdp_out = await asyncio.gather(*[
        pipeline_fsdp(Model(layers=4, batches=ranks, rank=i, dist=_dist))
        for i in range(16)])
    return (pipeline_fsdp_out,)


@app.cell
def _(Model, chalk, draw, pipeline_fsdp_out):
    Model.check(pipeline_fsdp_out)
    chalk.set_svg_height(1000)
    chalk.set_svg_draw_height(1000)

    draw(pipeline_fsdp_out)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### When does it make sense to combine?

    The goal of these exercises is to give you a sense of the different methods out there for distributed training. However, there is not currently a one size fits all approach for distributed training. The right choice will depend on the constants such as batch size, memory per GPU, communication overhead, implementation complexity, model size, and specifics of architecture.

    As an example  of what's left to explore, this last method Pipeline + FSDP is often not a great choice due to the complexities of communication speed. And in fact GPipe + FSDP also gets you into a bad place. The paper [Breadth First Pipeline Parallelism](https://arxiv.org/pdf/2211.05953.pdf) proposes instead a combination of pipeline scheduling and communication. Here's what it looks like.

    ![image.png](https://github.com/srush/LLM-Training-Puzzles/assets/35882/f286089a-83bd-483c-b441-f154821d161c)
    """)
    return


if __name__ == "__main__":
    app.run()
