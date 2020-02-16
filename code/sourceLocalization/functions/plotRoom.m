function p = plotRoom( roomSize, receivers, sourcesTr, sourcesTst,nL, sourcesTstPred )

%---------------------------------------------------------------
%plot all sources
%---------------------------------------------------------------
plotPosition= [50 50 1300 700];
sourcesTrL = sourcesTr(1:nL,:,:);
sourcesTrU = sourcesTr(nL+1:end,:,:);

if isempty(sourcesTst)
    for i = 1 : size(sourcesTrL, 1)
       p(2) = plot(sourcesTrL(i,1), sourcesTrL(i,2), 'Marker', 'o', 'MarkerSize', 10, 'MarkerEdgeColor', 'blue', 'MarkerFaceColor', 'red');
       L{2} = 'Labelled Tr.';
       hold on;
    end

    for i = 1 : size(sourcesTrU, 1)
       p(1) = plot(sourcesTrU(i,1), sourcesTrU(i,2), 'Marker', '.', 'MarkerSize', 15, 'MarkerEdgeColor', 'red', 'MarkerFaceColor', 'red');
       L{1} = 'Unlabelled Tr.';
       hold on;
    end
    %plot all receivers
    for i = 1 : size(receivers, 1)
       p(3) = plot(receivers(i,1), receivers(i,2), 'Marker', '+', 'MarkerSize', 10, 'MarkerEdgeColor', 'blue', 'MarkerFaceColor', 'blue');
       L{3} = 'Mic';
       hold on;
    end
else
    for i = 1 : size(sourcesTrL, 1)
       p(2) = plot(sourcesTrL(i,1), sourcesTrL(i,2), 'Marker', 'o', 'MarkerSize', 10, 'MarkerEdgeColor', 'blue', 'MarkerFaceColor', 'red');
       L{2} = 'Labelled Tr.';
       hold on;
    end
    for i = 1 : size(sourcesTrU, 1)
       p(1) = plot(sourcesTrU(i,1), sourcesTrU(i,2), 'Marker', '.', 'MarkerSize', 15, 'MarkerEdgeColor', 'red', 'MarkerFaceColor', 'red');
       L{1} = 'Unlabelled Tr.';
       hold on;
    end
    for i = 1 : size(sourcesTst, 1)
       p(3) = plot(sourcesTst(i,1), sourcesTst(i,2), 'Marker', 's', 'MarkerSize', 10,'MarkerEdgeColor', 'blue', 'MarkerFaceColor', 'blue');
       L{3} = 'Test';
       hold on;
    end
    for i = 1 : size(sourcesTstPred, 1)
       p(4) = plot(sourcesTstPred(i,1), sourcesTstPred(i,2), 'Marker', 's', 'MarkerSize', 10,'MarkerEdgeColor', 'blue', 'MarkerFaceColor', 'green');
       L{4} = 'Test Prediction';
       hold on;
    end

    %plot all receivers
    for i = 1 : size(receivers, 1)
       p(5) = plot(receivers(i,1), receivers(i,2), 'Marker', '+', 'MarkerSize', 10, 'MarkerEdgeColor', 'blue', 'MarkerFaceColor', 'blue');
       L{5} = 'Mic';
       hold on;
    end
end
%configure plot
title('room');
xlabel('x');
ylabel('y');
% zlabel('z');
legend(p,L);
% legend('Labelled Tr.', 'Unlabelled Tr.', 'Test', 'Mic')
grid on;
axis equal;
ax=gca;

ax.XLimMode = 'manual';
ax.YLimMode = 'manual';
ax.YLimMode = 'manual';

ax.XLim = [0, roomSize(1)];
ax.YLim = [0, roomSize(2)];
% ax.ZLim = [0, roomSize(3)];

% % insert labels
% sourceLabels = num2str([1 : num_arrays]);
% for i = 1:size(sources,1)
%     if mod(num_arrays,i) == 1
%         text(sources(i,1), sources(i,2), sources(i,3), sourceLabels(i), ...
%         'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
%     end
% end

% receiverLabels = cellstr( num2str([1 : size(receivers,1)]'));
% text(receivers(:,1), receivers(:,2), receiverLabels, ...
%     'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');

end

