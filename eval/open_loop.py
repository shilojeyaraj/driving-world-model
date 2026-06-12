"""Open-loop prediction eval: condition on k context frames + the TRUE action sequence,
roll the PRIOR forward, decode, compare predicted vs ground-truth future.

Mostly yours -- the metric is where world-model understanding shows.

Concept:  Prediction quality vs. horizon. Error grows with horizon; action-conditioning
          should help a lot (if it doesn't, you trained a video autoplayer, not dynamics).
Question: Why does action-conditioning matter for THIS metric specifically?
"""
def open_loop_eval(model, batch, context=5, horizon=20):
    # TODO:
    #   1. encode first `context` frames, rssm.observe -> a state
    #   2. rssm.imagine forward `horizon` steps using the TRUE actions
    #   3. decode predicted obs; compare to ground truth (MSE / SSIM / LPIPS for images)
    #   4. return error AS A FUNCTION OF horizon step -- plot it; the shape is the story
    raise NotImplementedError
