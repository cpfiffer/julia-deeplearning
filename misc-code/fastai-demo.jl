using FastAI, FastVision
data, blocks = load(datarecipes()["imagenette2-320"])
task = ImageClassificationSingle(blocks)
learner = tasklearner(task, data, callbacks=[ToGPU()])
fitonecycle!(learner, 10)
showoutputs(task, learner)