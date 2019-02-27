# Hashcode 2017

## Practice round

The code implements a solution by "guillotine cut" (see https://en.wikipedia.org/wiki/Cutting_stock_problem and http://tryalgo.org/fr/2017/01/22/google-hashcode-google-pizza/).
To speed up the solver, the input is processed by block, each block running in a separate process, and we simply merge the results of each block at the end.

This code will give a score of 1030301 in total.

## Qualification round

The code uses a greedy solution to select pair (vid=video id, cid=cache server id):
- each pair (vid, cid) is scored: we compute how much time we will win by adding this to the solution (and divide by the size of the video to take into account the fact that a bigger video will prevent putting more videos in the cache.
- we select the pair (vid, cid) with the best score
- we update the score for every remaining candidates: this part is fast enough because we don't need to re-score all pairs, only the ones impacted by the last choice.

It's possible to tune the weight put on the size of the video: gain = time_gain/pow(video_size, w).

Best results seems to be with w = 0.9

Score with this code: 2638034
Highest score during the Online Qualification Round was 2651999 (Team Ababahalamaha), and 2653781 (Team Master Exploder) in the Extended Round.
