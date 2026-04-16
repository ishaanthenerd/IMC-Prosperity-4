import { Button, Group, NumberInput } from '@mantine/core';
import Highcharts from 'highcharts';
import { ReactNode, useMemo, useState } from 'react';
import { ProsperitySymbol, Trade } from '../../models.ts';
import { useStore } from '../../store.ts';
import { getAskColor, getBidColor } from '../../utils/colors.ts';
import { formatNumber } from '../../utils/format.ts';
import { Chart } from './Chart.tsx';

export interface ProductPriceChartProps {
  symbol: ProsperitySymbol;
}

const STORAGE_KEY = 'imc-prosperity-3-viz-trade-point-cap';

function tradeVolume(point: Highcharts.Point): number {
  const opts = point.options as { custom?: { volume?: number } };
  return opts.custom?.volume ?? 0;
}

function normId(s: string | undefined): string {
  return (s ?? '').trim();
}

function isSubmission(s: string | undefined): boolean {
  return normId(s).toUpperCase() === 'SUBMISSION';
}

/** Evenly subsample sorted-by-time trades (inclusive endpoints). */
function evenSample<T>(sorted: T[], max: number): T[] {
  if (sorted.length <= max) {
    return sorted;
  }
  const out: T[] = [];
  const last = sorted.length - 1;
  for (let i = 0; i < max; i++) {
    const idx = Math.round((i / (max - 1)) * last);
    out.push(sorted[idx]);
  }
  return out;
}

function toScatterPoint(t: Trade) {
  return {
    x: t.timestamp,
    y: t.price,
    custom: { volume: t.quantity },
  };
}

function scatterSeries(
  name: string,
  color: string,
  label: string,
  data: Highcharts.PointOptionsType[],
): Highcharts.SeriesOptionsType {
  return {
    type: 'scatter',
    name,
    color,
    dataGrouping: { enabled: false },
    marker: {
      symbol: 'circle',
      radius: 5,
      lineWidth: 1,
      lineColor: '#ffffff',
    },
    tooltip: {
      pointFormatter: function (this: Highcharts.Point) {
        const v = tradeVolume(this);
        return `<span style="color:${this.color}">●</span> ${label}: <b>${formatNumber(v)}</b> @ <b>${formatNumber(this.y as number)}</b><br/>`;
      },
    },
    data,
  };
}

export function ProductPriceChart({ symbol }: ProductPriceChartProps): ReactNode {
  const algorithm = useStore(state => state.algorithm)!;

  const [capInput, setCapInput] = useState(() => {
    if (typeof window === 'undefined') {
      return 100;
    }
    const raw = window.localStorage.getItem(STORAGE_KEY);
    const n = raw == null ? 100 : Number.parseInt(raw, 10);
    return Number.isFinite(n) && n >= 1 ? n : 100;
  });
  const [capSaved, setCapSaved] = useState(capInput);

  const midByTs = useMemo(() => {
    const m = new Map<number, number>();
    for (const row of algorithm.activityLogs) {
      if (row.product !== symbol) {
        continue;
      }
      m.set(row.timestamp, row.midPrice);
    }
    return m;
  }, [algorithm.activityLogs, symbol]);

  const series = useMemo((): Highcharts.SeriesOptionsType[] => {
    const lineSeries: Highcharts.SeriesOptionsType[] = [
      { type: 'line', name: 'Bid 3', color: getBidColor(0.5), marker: { symbol: 'square' }, data: [] },
      { type: 'line', name: 'Bid 2', color: getBidColor(0.75), marker: { symbol: 'circle' }, data: [] },
      { type: 'line', name: 'Bid 1', color: getBidColor(1.0), marker: { symbol: 'triangle' }, data: [] },
      { type: 'line', name: 'Mid price', color: 'gray', dashStyle: 'Dash', marker: { symbol: 'diamond' }, data: [] },
      { type: 'line', name: 'Ask 1', color: getAskColor(1.0), marker: { symbol: 'triangle-down' }, data: [] },
      { type: 'line', name: 'Ask 2', color: getAskColor(0.75), marker: { symbol: 'circle' }, data: [] },
      { type: 'line', name: 'Ask 3', color: getAskColor(0.5), marker: { symbol: 'square' }, data: [] },
    ];

    for (const row of algorithm.activityLogs) {
      if (row.product !== symbol) {
        continue;
      }

      for (let i = 0; i < row.bidPrices.length; i++) {
        (lineSeries[2 - i] as any).data.push([row.timestamp, row.bidPrices[i]]);
      }

      (lineSeries[3] as any).data.push([row.timestamp, row.midPrice]);

      for (let i = 0; i < row.askPrices.length; i++) {
        (lineSeries[i + 4] as any).data.push([row.timestamp, row.askPrices[i]]);
      }
    }

    const ownBuy: Trade[] = [];
    const ownSell: Trade[] = [];
    const botBuy: Trade[] = [];
    const botSell: Trade[] = [];

    for (const row of algorithm.data) {
      const own = row.state.ownTrades[symbol];
      if (own) {
        for (const t of own) {
          if (isSubmission(t.buyer)) {
            ownBuy.push(t);
          } else if (isSubmission(t.seller)) {
            ownSell.push(t);
          } else {
            const mid = midByTs.get(t.timestamp);
            if (mid != null && Number.isFinite(mid) && t.price >= mid) {
              ownBuy.push(t);
            } else {
              ownSell.push(t);
            }
          }
        }
      }
      const mkt = row.state.marketTrades[symbol];
      if (mkt) {
        for (const t of mkt) {
          if (isSubmission(t.buyer) || isSubmission(t.seller)) {
            continue;
          }
          const mid = midByTs.get(t.timestamp);
          if (mid != null && Number.isFinite(mid) && t.price >= mid) {
            botBuy.push(t);
          } else {
            botSell.push(t);
          }
        }
      }
    }

    const cmp = (a: Trade, b: Trade) => a.timestamp - b.timestamp;
    ownBuy.sort(cmp);
    ownSell.sort(cmp);
    botBuy.sort(cmp);
    botSell.sort(cmp);

    const cap = capSaved;
    const ownBuyPts = evenSample(ownBuy, cap).map(toScatterPoint);
    const ownSellPts = evenSample(ownSell, cap).map(toScatterPoint);
    const botBuyPts = evenSample(botBuy, cap).map(toScatterPoint);
    const botSellPts = evenSample(botSell, cap).map(toScatterPoint);

    return [
      ...lineSeries,
      scatterSeries('Own (buy filled)', '#e03131', 'Own Trade (buy)', ownBuyPts),
      scatterSeries('Own (sell filled)', '#2f9e44', 'Own Trade (sell)', ownSellPts),
      scatterSeries('Bot (buy filled)', '#ff8787', 'bot trade (buy)', botBuyPts),
      scatterSeries('Bot (sell filled)', '#8ce99a', 'bot trade (sell)', botSellPts),
    ];
  }, [algorithm, symbol, midByTs, capSaved]);

  return (
    <>
      <Group justify="flex-end" gap="xs" mb="xs" wrap="wrap">
        <NumberInput
          label="Max markers / bucket"
          description="Evenly spaced over time; saved in this browser"
          size="xs"
          w={160}
          min={1}
          max={10_000}
          value={capInput}
          onChange={v => setCapInput(typeof v === 'number' ? v : Number.parseInt(String(v), 10) || 100)}
        />
        <Button
          size="xs"
          mt={22}
          onClick={() => {
            const n = Number.isFinite(capInput) && capInput >= 1 ? Math.floor(capInput) : 100;
            setCapInput(n);
            setCapSaved(n);
            if (typeof window !== 'undefined') {
              window.localStorage.setItem(STORAGE_KEY, String(n));
            }
          }}
        >
          Save
        </Button>
      </Group>
      <Chart title={`${symbol} - Price`} series={series} />
    </>
  );
}
