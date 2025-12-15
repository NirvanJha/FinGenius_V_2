import { LegendItem } from "../../../types";
interface UseDefaultLegendProps {
    items: LegendItem[];
    containerWidth?: number;
    buttonWidth?: number;
    isExpanded: boolean;
}
interface UseDefaultLegendResult {
    displayItems: LegendItem[];
    hasMoreItems: boolean;
    toggleButtonText: string;
}
export declare const useDefaultLegend: ({ items, containerWidth, buttonWidth, isExpanded, }: UseDefaultLegendProps) => UseDefaultLegendResult;
export {};
//# sourceMappingURL=useDefaultLegend.d.ts.map