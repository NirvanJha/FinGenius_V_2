import { ScatterChartData, ScatterPoint } from "../types";
/**
 * Extracts dataset names from scatter chart data
 * @param data - The scatter chart data (array of datasets)
 * @returns Array of dataset names
 */
export declare const getScatterDatasets: (data: ScatterChartData) => string[];
/**
 * Transforms scatter chart data for recharts consumption
 * @param data - The scatter chart data (array of datasets)
 * @param datasets - Array of dataset names to include
 * @param colors - Array of colors for datasets
 * @returns Flattened array of all points with color and dataset info
 */
export declare const transformScatterData: (data: ScatterChartData, datasets: string[], colors: string[]) => (ScatterPoint & {
    color: string;
    dataset: string;
})[];
/**
 * Calculates the domain for scatter chart axes
 * @param data - The scatter chart data (array of datasets)
 * @param axis - Which axis ('x' or 'y')
 * @returns Domain array [min, max] with padding
 */
export declare const calculateScatterDomain: (data: ScatterChartData, axis: "x" | "y") => [number, number];
/**
 * Formats scatter chart data for tooltip display
 * @param dataKey - The data key being displayed
 * @param value - The value to format
 * @param unit - Optional unit to append
 * @returns Formatted string
 */
export declare const formatScatterTooltipValue: (value: number | string, unit?: string) => string;
//# sourceMappingURL=ScatterChartUtils.d.ts.map