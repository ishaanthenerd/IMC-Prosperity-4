import { Center, Container, Grid, Stack, Table, Title } from '@mantine/core';
import { ReactNode } from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useStore } from '../../store.ts';
import { formatNumber } from '../../utils/format.ts';
import {
  formatSharpeRatio,
  periodPnLByTimestamps,
  sharpeFromPeriodPnL,
  sortedActivityTimestamps,
} from '../../utils/sharpe.ts';
import { AlgorithmSummaryCard } from './AlgorithmSummaryCard.tsx';
import { ConversionPriceChart } from './ConversionPriceChart.tsx';
import { EnvironmentChart } from './EnvironmentChart.tsx';
import { PlainValueObservationChart } from './PlainValueObservationChart.tsx';
import { PositionChart } from './PositionChart.tsx';
import { ProductPriceChart } from './ProductPriceChart.tsx';
import { ProfitLossChart } from './ProfitLossChart.tsx';
import { TimestampsCard } from './TimestampsCard.tsx';
import { TransportChart } from './TransportChart.tsx';
import { VisualizerCard } from './VisualizerCard.tsx';
import { VolumeChart } from './VolumeChart.tsx';

export function VisualizerPage(): ReactNode {
  const algorithm = useStore(state => state.algorithm);

  const { search } = useLocation();

  if (algorithm === null) {
    return <Navigate to={`/${search}`} />;
  }

  const conversionProducts = new Set();
  for (const row of algorithm.data) {
    for (const product of Object.keys(row.state.observations.conversionObservations)) {
      conversionProducts.add(product);
    }
  }

  let profitLoss = 0;
  const lastTimestamp = algorithm.activityLogs[algorithm.activityLogs.length - 1].timestamp;
  for (let i = algorithm.activityLogs.length - 1; i >= 0 && algorithm.activityLogs[i].timestamp == lastTimestamp; i--) {
    profitLoss += algorithm.activityLogs[i].profitLoss;
  }

  const symbols = new Set<string>();
  const plainValueObservationSymbols = new Set<string>();

  for (let i = 0; i < algorithm.data.length; i += 1000) {
    const row = algorithm.data[i];

    for (const key of Object.keys(row.state.listings)) {
      symbols.add(key);
    }

    for (const key of Object.keys(row.state.observations.plainValueObservations)) {
      plainValueObservationSymbols.add(key);
    }
  }

  const sortedSymbols = [...symbols].sort((a, b) => a.localeCompare(b));
  const sortedPlainValueObservationSymbols = [...plainValueObservationSymbols].sort((a, b) => a.localeCompare(b));

  const activityTimestamps = sortedActivityTimestamps(algorithm.activityLogs);
  const overallPeriodPnL = periodPnLByTimestamps(algorithm.activityLogs, activityTimestamps, 'total');
  const finalSharpeRatio = sharpeFromPeriodPnL(overallPeriodPnL);

  const sharpeRows: ReactNode[] = [
    <Table.Tr key="overall">
      <Table.Td>Overall</Table.Td>
      <Table.Td>{formatSharpeRatio(finalSharpeRatio)}</Table.Td>
    </Table.Tr>,
  ];
  for (const symbol of sortedSymbols) {
    const s = sharpeFromPeriodPnL(periodPnLByTimestamps(algorithm.activityLogs, activityTimestamps, symbol));
    sharpeRows.push(
      <Table.Tr key={symbol}>
        <Table.Td>{symbol}</Table.Td>
        <Table.Td>{formatSharpeRatio(s)}</Table.Td>
      </Table.Tr>,
    );
  }

  const symbolColumns: ReactNode[] = [];
  sortedSymbols.forEach(symbol => {
    symbolColumns.push(
      <Grid.Col key={`${symbol} - product price`} span={{ xs: 12, sm: 6 }}>
        <ProductPriceChart symbol={symbol} />
      </Grid.Col>,
    );

    symbolColumns.push(
      <Grid.Col key={`${symbol} - symbol`} span={{ xs: 12, sm: 6 }}>
        <VolumeChart symbol={symbol} />
      </Grid.Col>,
    );

    if (!conversionProducts.has(symbol)) {
      return;
    }

    symbolColumns.push(
      <Grid.Col key={`${symbol} - conversion price`} span={{ xs: 12, sm: 6 }}>
        <ConversionPriceChart symbol={symbol} />
      </Grid.Col>,
    );

    symbolColumns.push(
      <Grid.Col key={`${symbol} - transport`} span={{ xs: 12, sm: 6 }}>
        <TransportChart symbol={symbol} />
      </Grid.Col>,
    );

    symbolColumns.push(
      <Grid.Col key={`${symbol} - environment`} span={{ xs: 12, sm: 6 }}>
        <EnvironmentChart symbol={symbol} />
      </Grid.Col>,
    );

    symbolColumns.push(<Grid.Col key={`${symbol} - environment`} span={{ xs: 12, sm: 6 }} />);
  });

  sortedPlainValueObservationSymbols.forEach(symbol => {
    symbolColumns.push(
      <Grid.Col key={`${symbol} - plain value observation`} span={{ xs: 12, sm: 6 }}>
        <PlainValueObservationChart symbol={symbol} />
      </Grid.Col>,
    );
  });

  return (
    <Container fluid>
      <Grid>
        <Grid.Col span={12}>
          <VisualizerCard>
            <Center>
              <Stack gap="xs" align="center">
                <Title order={2}>Final Profit / Loss: {formatNumber(profitLoss)}</Title>
                <Title order={2}>Final Sharpe Ratio: {formatSharpeRatio(finalSharpeRatio)}</Title>
              </Stack>
            </Center>
          </VisualizerCard>
        </Grid.Col>
        <Grid.Col span={12}>
          <VisualizerCard title="Sharpe ratio (per timestamp PnL)">
            <Table.ScrollContainer minWidth={300}>
              <Table withColumnBorders horizontalSpacing={8} verticalSpacing={4}>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>Product</Table.Th>
                    <Table.Th>Sharpe ratio</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>{sharpeRows}</Table.Tbody>
              </Table>
            </Table.ScrollContainer>
          </VisualizerCard>
        </Grid.Col>
        <Grid.Col span={{ xs: 12, sm: 6 }}>
          <ProfitLossChart symbols={sortedSymbols} />
        </Grid.Col>
        <Grid.Col span={{ xs: 12, sm: 6 }}>
          <PositionChart symbols={sortedSymbols} />
        </Grid.Col>
        {symbolColumns}
        <Grid.Col span={12}>
          <TimestampsCard />
        </Grid.Col>
        {algorithm.summary && (
          <Grid.Col span={12}>
            <AlgorithmSummaryCard />
          </Grid.Col>
        )}
      </Grid>
    </Container>
  );
}
