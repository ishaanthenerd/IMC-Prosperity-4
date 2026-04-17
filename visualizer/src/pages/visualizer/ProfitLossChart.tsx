import Highcharts from 'highcharts';
import { ReactNode } from 'react';
import { useStore } from '../../store.ts';
import { Chart } from './Chart.tsx';

export interface ProfitLossChartProps {
  symbols: string[];
}

export function ProfitLossChart({ symbols }: ProfitLossChartProps): ReactNode {
  const algorithm = useStore(state => state.algorithm)!;

  const dataByTimestamp = new Map<number, number>();
  for (const row of algorithm.activityLogs) {
    if (!dataByTimestamp.has(row.timestamp)) {
      dataByTimestamp.set(row.timestamp, row.profitLoss);
    } else {
      dataByTimestamp.set(row.timestamp, dataByTimestamp.get(row.timestamp)! + row.profitLoss);
    }
  }

  const sortedTimestamps = [...dataByTimestamp.keys()].sort((a, b) => a - b);

  const series: Highcharts.SeriesOptionsType[] = [
    {
      type: 'line',
      name: 'Total',
      data: sortedTimestamps.map(timestamp => [timestamp, dataByTimestamp.get(timestamp)]),
    },
  ];

  symbols.forEach(symbol => {
    const data = [];

    for (const row of algorithm.activityLogs) {
      if (row.product === symbol) {
        data.push([row.timestamp, row.profitLoss]);
      }
    }

    series.push({
      type: 'line',
      name: symbol,
      data,
      dashStyle: 'Dash',
    });
  });

  return <Chart title="Profit / Loss" series={series} />;
}
