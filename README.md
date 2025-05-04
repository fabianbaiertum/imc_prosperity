# imc_prosperity

In here, I got the code I wrote for imc prosperity3 for the first round and some other ideas which might be useful for future trading competitions. I also add some functions, where I simply have the trading logic implemented, which was only 
tested, but not used in our final submission (such as the RSI_bet_sizing.py).

In round 1, our team was ranked 19 (top 0.15 percent of participants), but a few of the contestants in front of us were reset to 0 points due to hardcoding their answers.

For cubic_atm_spline.py I just wanted to test if we interpolate the implied volatilities for the different strikes to estimate the implied vol. ATM at each time step, if we could find a pattern for it, which we could use for a trading strategy, it didn't work too well.

RSI_bet_sizing.py tries to implement the bet sizing in case we run an RSI based strategy, with focus on having a higher bet size when we are more confident, i.e. if RSI>90: we are highly confident or if it is RSI<10: we are also highly confident probabilitically speaking, given a specific distribution.

I'm also only uploading my ideas, not the ones from my teammates. I won't share my final code as it might impact future trading competitions.

bollingerbands.py was one of the first iterations for INK, also includes Taking, Clearing and MM logic, but a simplified version of it.


The strategy for Taking and Clearing was quite simple, we just buy when the offered price is below a certain amount of our theoretical value and for clearing we try to close positions which are close the EV=0 when we can unwind our position.

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

I also tried the Guilbaud & Pham Framework.
In the future, I should also try to forecast the theoretical value, if the competition allows for it.

### Strategy for INK (or in general prediction/forecasting a highly volatile asset)

For INK I tried every simple strategy, from ARIMA to any moving averages (simple, exponential) or oscillator strategies (Bollinger bands, crossing moving averages, RSI). On the original data RSI and bollinger bands worked quite well,
but I also tested a z-score based version of mean reversion, which worked a little bit better. But after round 2 they changed the distribution of the data set since a few teams cheated of using their knowledge of last years data set distribution, so I had to retry those methods, and now EMA worked. I hope for the next year they will allow that we can use deep learning libraries and also import our models.

### Strategy for baskets

There were 3 possible stat arb scenarios, we used all of them, but for simplicity we put all the liquidity in the basket 1 vs. the three components of this basket, then what was left of liquidity in basket 2 vs. basket 2 components and then with the remaining volume we put it into our combination of them (something like 2 Basket2 - 1 Basket1 +..+... = some combination of components). I used something similar to the book: Algorithmic Trading and Quantitative Strategies by Velu, Hardy and Nehren for their pairs trading algorithm on page 176. 
My strategy was to first again just calculate theo. value for the basket1 and all of the components using my microprice calculations with the LOB as input. Then I compare 
price_basket1-price_basket_1_components, e.g. let basket1 be 2 shares of apple stock and 3 shares of tesla stock, then price basket 1 components is 2*price_apple+3*price_tesla.
If this difference is larger as epsilon+ some estimated spread crossing costs (e.g. just summing all the spreads up) basket1 would be overpriced in this case. Thus we short basket1 and go long in the components in appropriate proportions.
We must take the minimum of our available position limits, to not breach anything. If the price mean reverts, we close the position and wait until either the basket or its components gets overpriced.
For position sizing, the bigger the divergence of the assets, the larger the position should be.


### Strategy for options on the same underlying with different strikes

I used the basic BS model to price my options, the interest rate was set to r=0, for all other inputs, I knew the values, besides the implied volatility.
The strike was obviously known, maturity had to be recalculated every time step, so T=7 at t=0, then at t=1 T-time_step_size. For the underlying price I just used my microprice function using the LOB of the underlying.
So the last challenge was to estimate the implied volatility, for it I used the bisection method, so I input different imp. vol. values until I'm close enough to the microprice quoted on the market. 

My first strategy was just delta hedging my positions when I find an undervalued/overvalued option. I would calculate the total delta of the options, then decide which ones are more reasonable (larger edge) and then hedge the trades I would like to take with the underlying with -delta amount of units, and take the minimum that no position limit is breached. To keep my delta hedging active, I recalculate the delta every time step and adjust my positions. This simple strategy alone made more profits than all previous strategies combined, but the PnL was highly volatile, so I also included delta-gamma hedging since the underlying was too volatile and thus we also need to consider gamma. 
For delta-gamma hedging I split the options into groups, the ones which I buy and the ones which I gamma hedge with. Also, for T going to zero, i.e. close to expiry of the options, I don't want to be negatively effected by the time decay.
Thus close to expiry, I only want to short options (which are OTM) or take long options with a large enough edge and also I want to close them as soon as possible. 


## To fix for upcoming competitions

Some major issues in the competition were: scalability and I used too many parameters, which needed to be estimated. In an attempt to solve the scalability issue, I tried to create generic functions for the taking, clearing and market making for all assets traded, which was successful, but for compatability reasons, I didn't include it in our teams code since most of the new assets were only using new trading strategies, like statistical arbitrage for the basket and something like delta-gamma hedging for the options, but when developing an algorithm for a huge system of assets, generic functions with as little as possible parameters, which can be adaptively calculated, would be the goal.
For future competitions, I should also include vega hedging to stay vega neutral (flat on volatility) or to forecast implied vol. to decide if I should be vega positive or negative in the current market.

For market making:      I also tried the Guilbaud & Pham Framework.
                        In the future, I should also try to forecast the theoretical value, if the competition allows for it.
