import { BarChartVariant } from "../../types";
export declare const getRadiusArray: (variant: BarChartVariant, radius: number, orientation: "vertical" | "horizontal", isFirst?: boolean, isLast?: boolean, isNegative?: boolean) => [number, number, number, number];
export declare const findNearestSnapPosition: (snapPositions: number[], currentScroll: number, direction: "up" | "down" | "left" | "right") => number;
export interface BarStackInfo {
    isNegative: boolean;
    isFirstInStack?: boolean;
    isLastInStack?: boolean;
    hasNegativeValueInStack?: boolean;
}
export declare function getBarStackInfo(variant: "grouped" | "stacked", value: number | [number, number], dataKey: string, payload: Record<string, unknown>, dataKeys: string[]): BarStackInfo;
//# sourceMappingURL=BarChartsUtils.d.ts.map