# imc_prosperity

In here, I got the code I wrote for imc prosperity3 for the first round and some other ideas which might be useful for future trading competitions. I also add some functions, where I simply have the trading logic implemented, which was only 
tested, but not used in our final submission (such as the RSI_bet_sizing.py).

In round 1, our team was ranked 19, but a few of the contestants in front of us were reset to 0 points due to hardcoding their answers.

For cubic_atm_spline.py I just wanted to test if we interpolate the implied volatilities for the different strikes to estimate the implied vol. ATM at each time step, if we could find a pattern for it, which we could use for a trading strategy, it didn't work too well.

RSI_bet_sizing.py tries to implement the bet sizing in case we run an RSI based strategy, with focus on having a higher bet size when we are more confident, i.e. if RSI>90: we are highly confident or if it is RSI<10: we are also highly confident probabilitically speaking, given a specific distribution.

I'm also only uploading my ideas, not the ones from my teammates. I won't share my final code as it might impact future trading competitions.




#### To fix for upcoming competitions

Some major issues in the competition were: scalability and I used too many parameters, which needed to be estimated. In an attempt to solve the scalability issue, I tried to create generic functions for the taking, clearing and market making for all assets traded, which was successful, but for compatability reasons, I didn't include it in our teams code since most of the new assets were only using new trading strategies, like statistical arbitrage for the basket and something like delta-gamma hedging for the options, but when developing an algorithm for a huge system of assets, generic functions with as little as possible parameters, which can be adaptively calculated, would be the goal.


### Strategy for Market Making

My market making strategy was quite simple:
1.Step: I'm calculating a theoretical value of the asset based on the current LOB only (so no forecasting of the theo. value), 
        using liquidity (volume) and bid-ask prices at all available levels with my own microprice calculations.
        In my microprice calculation, I give values closer to the level 1 microprice (only using top of the book information of volume and bid-ask prices) higher weights than values further away (so it is distance based).
        
2.Step: I'm calculating order flow imbalance to see if we should shift our bid and ask prices up or down.

3.Step: I calculate some volatility estimate to adjust how much my bid-ask prices should be spread.

4.Step: Now I also want to incorporate that as an MM I don't want to be directionally trading, thus I'm trying to be market neutral, so if I got a huge position in one direction, I would like to unwind it (hopefully without too much 
        market impact). Thus, I also introduce an inventory skew factor.

5.Step: Position sizing. As a market maker I always want to be trading the same lot size on both sides, otherwise I give the competition information in which direction I want to skew my prices. 

I also tried the Guilbaud & Pham Framework  
