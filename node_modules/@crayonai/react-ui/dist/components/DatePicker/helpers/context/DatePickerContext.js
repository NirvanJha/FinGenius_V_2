import { jsx as _jsx } from "react/jsx-runtime";
import { createContext, useContext } from "react";
const defaultContext = {
    selectedDate: undefined,
    selectedRange: { from: undefined, to: undefined },
    isOpen: false,
    mode: "single",
    botType: "fullscreen",
    setSelectedDate: (_date) => { },
    setSelectedRange: (_range) => { },
    setIsOpen: (_isOpen) => { },
};
const DatePickerContext = createContext(defaultContext);
export const useDatePicker = () => {
    const context = useContext(DatePickerContext);
    if (!context) {
        throw new Error("useDatePicker must be used within a DatePickerProvider");
    }
    return context;
};
export const DatePickerProvider = ({ children, selectedDateFromParent, selectedRangeFromParent, setSelectedDateFromParent, setSelectedRangeFromParent, isOpenFromParent, setIsOpenFromParent, mode, botType, }) => {
    return (_jsx(DatePickerContext.Provider, { value: {
            // DatePicker State needed from parent
            selectedDate: selectedDateFromParent,
            selectedRange: selectedRangeFromParent,
            setSelectedDate: setSelectedDateFromParent,
            setSelectedRange: setSelectedRangeFromParent,
            mode,
            isOpen: isOpenFromParent,
            setIsOpen: setIsOpenFromParent,
            botType,
        }, children: children }));
};
export default DatePickerContext;
//# sourceMappingURL=DatePickerContext.js.map