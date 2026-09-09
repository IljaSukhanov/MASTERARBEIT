# Preliminary mean-width results

Main run: T=2,000, M=8,192, k=1,...,10, uniform [-1,1] innovations.
The three thesis process seeds and zero-based replication indices were reproduced.

- AR(1): empirical mean width 3.742 at k=1 and 4.218 at k=10; population reference 4.582 and 6.915. At k=10 the paired support gap is 39.0% and the conditional MC SE is 0.0084.
- AR(2): empirical mean width 5.666 at k=1 and 5.885 at k=10; population reference 8.134 and 10.127. At k=10 the paired support gap is 41.5% and the conditional MC SE is 0.0175.
- AR(3): empirical mean width 5.507 at k=1 and 5.686 at k=10; population reference 13.980 and 15.175. At k=10 the paired support gap is 62.2% and the conditional MC SE is 0.0139.

**Sample-size experiment (new paths, fixed thesis coefficients):**

- AR(1), k=10: average paired gap decreases from 43.6% at T=500 to 31.9% at T=16,000.
- AR(2), k=10: average paired gap decreases from 50.6% at T=500 to 40.1% at T=16,000.
- AR(3), k=10: average paired gap decreases from 66.2% at T=500 to 54.1% at T=16,000.

Increasing M reduces direction-integration error; it does not fill the missing population support. The raw mean-width curves do not inherit the minimum-width plateau and do not yet define an order selector.

The largest change in population mean width when doubling H=2,048 was 0. This is a numerical diagnostic, not a certified tail bound.
