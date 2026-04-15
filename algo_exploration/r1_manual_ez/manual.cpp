#include <bits/stdc++.h>
using namespace std;

// map<int, int, greater<int>> bids {{30, 30000}, {29, 5000}, {28, 12000}, {27, 28000}};
// map<int, int> asks {{28, 40000}, {31, 20000}, {32, 20000}, {33, 30000}};
// constexpr int value = 30;
// constexpr int fee = 0;

map<int, int, greater<int>> bids {{20, 43000}, {19, 17000}, {18, 6000}, {17, 5000}, {16, 10000}, {15, 5000}, {14, 10000}, {13, 7000}};
map<int, int> asks {{12, 20000}, {13, 25000}, {14, 35000}, {15, 6000}, {16, 5000}, {17, 0}, {18, 10000}, {19, 12000}};
constexpr int value = 20;
constexpr int fee = 10;

int main() {
	int lower = asks.begin()->first, upper = bids.begin()->first;
	int minVolume = 0;
	for (int clearing = lower; clearing < upper; clearing++) {
		int volAsk = 0;
		for (auto [price, vol] : asks) {
			if (price > clearing)
				break;
			volAsk += vol;
		}
		int volBid = 0;
		for (auto [price, vol] : bids) {
			if (price < clearing)
				break;
			volBid += vol;
		}
		minVolume = max(minVolume, min(volAsk, volBid));
	}
	cout << "Default volume is " << minVolume << endl;

	int bestVolume = 0, bestProfit = 0;
	int bestClearing = 0, bestAmount = 0;

	for (int clearing = lower; clearing < upper; clearing++) {
		int volAsk = 0;
		for (auto [price, vol] : asks) {
			if (price > clearing)
				break;
			volAsk += vol;
		}

		int volBid = 0, lastBidVol = 0;
		for (auto [price, vol] : bids) {
			if (price < clearing)
				break;
			if (price == clearing) {
				lastBidVol = vol;
				break;
			}
			volBid += vol;
		}

		// if volBid >= volAsk, we cannot reach the clearing price bid
		if (volBid >= volAsk)
			continue;
		// must be forced to use last bid vol
		if (lastBidVol == 0)
			continue;
		// cannot trade less than the default volume
		if (volAsk < minVolume)
			continue;
		
		// volBid < volAsk, we can reach the clearing price for bids
		// must leave 1 at clearing price for bids, so use remaining
		int totalVolume = volAsk;
		int amount = volAsk - (volBid + 1);
		int profit = 100 * ((value - clearing) * amount) - fee * amount;
		printf("%d trades %d total volume, %d volume for us, makes %d.%02d profit\n",
			clearing, totalVolume, amount,
			profit / 100, profit % 100);

		if (profit > bestProfit) {
			bestVolume = totalVolume;
			bestProfit = profit;
			bestClearing = clearing;
			bestAmount = amount;
		}
	}

	cout << "Best is " << bestClearing << " with " << bestAmount << endl;
	printf("%d trades %d total volume, %d volume for us, makes %d.%02d profit\n",
		bestClearing, bestVolume, bestAmount,
		bestProfit / 100, bestProfit % 100);

}
