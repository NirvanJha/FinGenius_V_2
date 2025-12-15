import { AreaChartData } from "../../AreaChart/types";
import { LineChartData } from "../../LineChart/types";
type ChartData = AreaChartData | LineChartData;
/**
 * AREA CHART AND LINE CHART SPECIFIC FUNCTION
 * This function returns the width of the data in the area chart, used for padding calculation, scroll amount calculation, and
 * for the width of the chart container.
 * @param data - The data to be displayed in the chart.
 */
export declare const getWidthOfData: (data: ChartData, containerWidth: number) => number;
/**
 * SHARED UTILITY FUNCTION
 * This function returns the nearest snap position index for both chart types.
 * The implementation is identical for both AreaChart and LineChart.
 * @param snapPositions - The snap positions for the chart.
 * @param currentScroll - The current scroll of the chart.
 * @param direction - The direction of the scroll.
 * @returns The nearest snap position index for the chart.
 */
export declare const findNearestSnapPosition: (snapPositions: number[], currentScroll: number, direction: "left" | "right") => number;
/**
 * SHARED UTILITY FUNCTION
 * This function returns the width of each group/category for both chart types.
 * Both AreaChart and LineChart use the same ELEMENT_SPACING.
 * @param data - The data to be displayed in the chart.
 * @returns The width of each group/category.
 */
export declare const getWidthOfGroup: (data: ChartData) => 200 | 72;
/**
 * SHARED UTILITY FUNCTION
 * This function returns the snap positions for both chart types, used for smooth scrolling.
 * @param data - The data to be displayed in the chart.
 * @returns The snap positions for the chart.
 */
export declare const getSnapPositions: (data: ChartData) => number[];
export {};
//# sourceMappingURL=AreaAndLineUtils.d.ts.map