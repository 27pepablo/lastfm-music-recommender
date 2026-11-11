# How it works

A walk through the method, from the raw log to a recommendation, with the reasoning
behind each decision. The code for every step is in [`src/musicrec/`](../src/musicrec).

## 1. What the data says, and what it does not

The Last.fm log is a list of facts of the form *"user U played track T by artist A at
time X"*. Nobody rated anything. This is **implicit feedback**, and it differs from
ratings in three ways that shape the whole project:

1. **There is no negative signal.** A user who never played an artist may dislike
   them or may simply never have come across them. The data cannot tell the two apart.
2. **The numbers are counts, not opinions.** 500 plays means "likes a lot", but not
   "likes 250 times more than an artist with 2 plays".
3. **Some plays are not real listens.** A track that was skipped after ten seconds
   still shows up as a play.

## 2. Skips: reading intent from timestamps

The log has no "skipped" flag and no track durations. What it does have is the time of
every play, so the time until the same user's *next* play is an upper bound on how long
a track was heard. Half of all plays are followed by the next one within four minutes,
which is the length of a song; a small group is followed within seconds.

Each play gets a weight (`prepare.skip_weight`):

| Time to next play | Reading | Weight |
|---|---|---|
| under 30 s | skipped | 0 |
| 30 to 60 s | brief partial listen | 0.5 |
| 60 to 90 s | partial listen | 0.8 |
| 90 s or more, or last play | listened | 1 |

The *weighted play count* of a user-artist pair is the sum of these weights. The gaps
are measured on the complete log, before any artist is filtered out; measuring them
afterwards would make a skipped track look like a long listen whenever the following
play was removed.

## 3. One artist, one row

About 3 % of plays carry an artist name but no MusicBrainz identifier, and the same
artist often appears both ways. Plays without an identifier are attached to the
most-played identifier with the same name (ignoring case and spacing); when there is
none, the name itself is the key (`prepare.canonical_artists`). Artists played by
fewer than 5 users and users with fewer than 50 artists are dropped, because nothing
reliable can be learned about them.

## 4. Splitting by time

A recommender is used to predict what comes next, so it is tested that way. Each
user's plays are sorted by time and cut into three periods:

```
 first 70 % of plays      next 10 %      last 20 %
|------------------------|-------------|------------------|
          fit              validation          test
```

- Hyperparameters are chosen by training on *fit* and checking on *validation*.
- The reported numbers come from training on *fit + validation* and checking on *test*,
  which is used once.

A random split would let the model see a user's future while predicting their past.

## 5. What counts as a good recommendation

The recommender is asked for ten artists the user **has not played yet**. A
recommendation is a hit if the user plays that artist in the held-out period. Artists
the user already knows are excluded from both the recommendations and the ground
truth: suggesting someone's favourite band is easy and useless.

Metrics, all at ten recommendations (`evaluation.py`):

- **Precision@10**: share of the ten that were hits.
- **Recall@10**: share of the user's new artists that were found.
- **NDCG@10**: like precision, but a hit in first position counts more than one in tenth.
- **Hit rate**: share of users with at least one hit.
- **Coverage**: share of the catalogue that is recommended to at least one user.
- **Median audience**: how many listeners a typical recommended artist has.

The first four measure accuracy; the last two measure whether the system only ever
recommends the same famous names. Accuracy numbers look low in absolute terms because
the task is hard: ten guesses, 22 000 candidates, and a typical user picks up a few
dozen new artists in the test period. The useful comparison is against the baselines.

Reconstruction error is not reported. A model can reproduce play counts well and still
put the wrong ten artists on top.

## 6. From play counts to confidence

The models do not use the weighted play count directly. Three transformations are
compared (`weighting.py`):

- **linear**: use the count as is. One fan with 5 000 plays weighs as much as 5 000
  casual listeners.
- **log**: `log(1 + count)`. Going from 1 to 10 plays matters as much as from 10 to 100.
- **BM25**: borrowed from text search, with users as documents and artists as words.
  The weight saturates quickly with the count, is normalised by how much the user
  listens overall, and is multiplied by how *rare* the artist is. Playing an obscure
  band says more about your taste than playing the biggest band in the dataset.

## 7. The models

**Popularity.** Recommend the artists with the most listeners that the user has not
played. Not personalised, and hard to beat: popular artists are popular because many
people end up listening to them.

**Item-kNN.** Two artists are similar if the same users play both (cosine similarity
between their columns of the weighted matrix). A user's score for a new artist is the
similarity between it and the artists the user already plays. Simple, and a strong
baseline on small datasets.

**Implicit ALS** (Hu, Koren and Volinsky, 2008; `models.ImplicitALS`). Every user $u$
gets a vector $x_u$ and every artist $i$ a vector $y_i$ of $f$ numbers, chosen so that
their dot product is close to 1 when the user plays the artist and close to 0 when they
do not:

$$
\min_{x, y} \sum_{u, i} w_{ui}\,\bigl(p_{ui} - x_u^\top y_i\bigr)^2
\;+\; \lambda \Bigl(\sum_u \lVert x_u \rVert^2 + \sum_i \lVert y_i \rVert^2\Bigr),
\qquad
p_{ui} = \begin{cases} 1 & \text{played} \\ 0 & \text{not played} \end{cases},
\qquad
w_{ui} = 1 + \alpha\, c_{ui}.
$$

$c_{ui}$ is the confidence from the previous section (zero for pairs never played).
Every user-artist pair appears in the sum, including the 97 % that were never played;
they enter with weight 1 and target 0. That is how the model deals with the missing
negative signal: "not played" is weak evidence of "not interested", and "played a lot"
is strong evidence of "interested".

*Alternating least squares.* With the artist vectors fixed, the best vector for each
user solves a small linear system:

$$
\bigl(Y^\top Y + \lambda I + Y_S^\top \mathrm{diag}(\alpha c_u)\, Y_S\bigr)\, x_u = Y_S^\top (1 + \alpha c_u),
$$

where $Y_S$ holds the vectors of the artists the user has played. $Y^\top Y$ is the same
for all users and is computed once, so the cost depends on the number of observed
pairs and not on the 21 million cells of the full matrix. The same holds with the roles
swapped, and the algorithm alternates between the two sides.

*Solver.* Instead of factorising one matrix per user, all the systems of one side are
advanced together with three conjugate-gradient steps, starting from the previous
vectors (Takács et al., 2011). One step needs the product $A_u p_u$ for every row,
which is two sparse matrix products with no Python loop. With 30 steps the result
matches the exact solution to the last printed digit; the test suite checks this.

The three hyperparameters ($f$, $\lambda$, $\alpha$) are chosen by random search on the
validation period, separately for each weighting.

## 8. Using the artist vectors

After training, artists that share listeners have vectors pointing in similar
directions. This gives three things for free (`explore.py`):

- **Similar artists**: nearest neighbours by cosine similarity.
- **Hidden gems**: the nearest neighbours among artists with a small audience
  (8 to 25 listeners), skipping entries that are a duplicate spelling of a famous name.
- **A map**: project the vectors to two dimensions with t-SNE.

No genre, tag or audio feature is used anywhere. Any structure in the map comes from
listening behaviour alone.

## 9. Popularity bias, and a knob to control it

Recommenders tend to over-recommend what is already popular. Dividing each score by
$(\text{listeners} + 1)^\beta$ before ranking counters this: $\beta = 0$ changes
nothing, larger values favour artists with smaller audiences. The README shows what
this costs in accuracy and buys in variety.

## References

- Y. Hu, Y. Koren, C. Volinsky. *Collaborative Filtering for Implicit Feedback Datasets.* ICDM 2008.
- G. Takács, I. Pilászy, D. Tikk. *Applications of the Conjugate Gradient Method for Implicit Feedback Collaborative Filtering.* RecSys 2011.
- S. Robertson, H. Zaragoza. *The Probabilistic Relevance Framework: BM25 and Beyond.* 2009.
- Ò. Celma. *Music Recommendation and Discovery in the Long Tail.* Springer, 2010 (source of the dataset).
