import React from "react";
import { DateRange } from "react-day-picker";
interface DatePickerContextType {
    selectedDate: Date | undefined;
    selectedRange: DateRange | undefined;
    isOpen: boolean;
    mode: "single" | "range";
    botType: "mobile" | "fullscreen" | "tray" | "copilot";
    setSelectedDate: (date: Date | undefined) => void;
    setSelectedRange: (range: DateRange | undefined) => void;
    setIsOpen: (isOpen: boolean) => void;
}
declare const DatePickerContext: React.Context<DatePickerContextType>;
export declare const useDatePicker: () => DatePickerContextType;
interface DatePickerProviderProps {
    children: React.ReactNode;
    selectedDateFromParent: Date | undefined;
    selectedRangeFromParent: DateRange | undefined;
    mode: "single" | "range";
    botType: "mobile" | "fullscreen" | "tray" | "copilot";
    setSelectedDateFromParent: (date: Date | undefined) => void;
    setSelectedRangeFromParent: (range: DateRange | undefined) => void;
    isOpenFromParent: boolean;
    setIsOpenFromParent: (isOpen: boolean) => void;
}
export declare const DatePickerProvider: React.FC<DatePickerProviderProps>;
export default DatePickerContext;
//# sourceMappingURL=DatePickerContext.d.ts.map